"""workflow/streaming.py — Incremental extraction of one JSON string field
from a *partial* token stream.

WHY THIS EXISTS
---------------
The supervisor runs with `format="json"` and packs the user-facing answer
into a `"final_response"` field alongside its routing decision:

    {"route": "critic", "tasks": [], "final_response": "## Answer\\n\\n..."}

That is the right shape for the graph — routing and answer decided in one
call — but it means the answer is NOT the model's raw output stream, so we
cannot simply forward tokens to the browser. `json.loads` needs the whole
object, which is exactly the thing we are trying not to wait for.

This class walks the raw token stream as it arrives, finds the opening
quote of the target field, and decodes JSON string escapes incrementally,
returning only the characters that became available since the last call.
Net effect: the user starts reading the answer while the model is still
writing it, instead of after the full object closes.

DESIGN NOTES
------------
- Escape sequences can straddle a chunk boundary ("...\\" then "n..."), so
  the scanner stops and *waits* on an incomplete escape rather than
  emitting a stray backslash. Position is only advanced past bytes that
  were fully decoded.
- On planning loops the model emits `"final_response": null`. The opening
  pattern requires a quote, so it simply never matches and nothing is
  emitted — no special-casing needed at the call site.
- This is a best-effort accelerator, never the source of truth. The
  authoritative answer is still whatever the node returns in graph state;
  the endpoint re-sends the complete text when the node ends. If this
  class extracts nothing (model didn't stream, field ordering surprise,
  malformed escape), the UI simply shows the answer a moment later, the
  way it did before streaming existed. It cannot corrupt the final result.
"""
import re
from typing import Optional

# Matches the field name through the opening quote of its string value,
# tolerating the whitespace variations different models emit.
_OPEN_RE_TEMPLATE = r'"{field}"\s*:\s*"'

# Single-character JSON escapes. Anything not listed decodes to itself,
# which matches how lenient JSON readers behave and avoids throwing away
# text over a malformed escape mid-stream.
_SIMPLE_ESCAPES = {
    "n": "\n",
    "t": "\t",
    "r": "\r",
    "b": "\b",
    "f": "\f",
    '"': '"',
    "\\": "\\",
    "/": "/",
}


class PartialJSONFieldStreamer:
    """Feed raw model tokens in; get decoded characters of one field out.

    Usage:
        s = PartialJSONFieldStreamer("final_response")
        for token in stream:
            delta = s.feed(token)
            if delta:
                send_to_browser(delta)
    """

    def __init__(self, field: str = "final_response") -> None:
        self._open_re = re.compile(_OPEN_RE_TEMPLATE.format(field=re.escape(field)))
        self.reset()

    def reset(self) -> None:
        """Clear all state. Called between supervisor loops.

        The supervisor may run several times in one turn (plan → dispatch
        workers → plan again → answer). Each invocation is a fresh JSON
        object, so carrying a buffer across them would try to decode the
        new object's bytes as a continuation of the old one's string.
        """
        self._buf = ""
        self._pos: Optional[int] = None   # cursor inside the field's value
        self._done = False                # closing quote consumed
        self._emitted = 0                 # decoded chars handed out so far

    @property
    def started(self) -> bool:
        """True once the target field's opening quote has been seen."""
        return self._pos is not None

    @property
    def done(self) -> bool:
        """True once the field's closing quote has been consumed."""
        return self._done

    @property
    def emitted_chars(self) -> int:
        """How much decoded text this streamer has handed out."""
        return self._emitted

    def feed(self, chunk: str) -> str:
        """Append raw stream text; return newly decoded field characters.

        Returns "" when there is nothing new to show yet — which is the
        common case for every token before the field starts, and for a
        chunk that ends mid-escape.
        """
        if not chunk or self._done:
            return ""

        self._buf += chunk

        # Not inside the value yet: look for the field's opening quote.
        if self._pos is None:
            match = self._open_re.search(self._buf)
            if not match:
                return ""
            self._pos = match.end()

        out = []
        i = self._pos
        buf = self._buf
        n = len(buf)

        while i < n:
            c = buf[i]

            if c == "\\":
                # Incomplete escape at the tail of the buffer: stop here and
                # leave the cursor *before* the backslash so the next feed()
                # re-examines it with more bytes available.
                if i + 1 >= n:
                    break
                esc = buf[i + 1]
                if esc == "u":
                    # \uXXXX needs four hex digits; wait if they haven't landed.
                    if i + 6 > n:
                        break
                    hex4 = buf[i + 2:i + 6]
                    try:
                        out.append(chr(int(hex4, 16)))
                    except ValueError:
                        # Not valid hex — pass it through literally rather
                        # than dropping characters the user should see.
                        out.append(buf[i:i + 6])
                    i += 6
                    continue
                out.append(_SIMPLE_ESCAPES.get(esc, esc))
                i += 2
                continue

            if c == '"':
                # Unescaped quote terminates the string value.
                self._done = True
                i += 1
                break

            out.append(c)
            i += 1

        self._pos = i
        text = "".join(out)
        self._emitted += len(text)
        return text
