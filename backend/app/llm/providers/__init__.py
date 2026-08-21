"""One module per vendor, each one an adapter and nothing more.

A provider here implements `app.llm.base.Provider`: it turns our four calls into
one vendor's request shape and that vendor's exceptions into our `ErrorClass`
names. Fallback chains, circuit breakers, retries, timeouts and cost accounting
are not its business — they live in `app.llm.router` and `app.llm.usage`, once,
for every vendor.

Three modules live here — `openai_provider`, `gemini_provider` and
`anthropic_provider` — behind the prefixes ``openai:``, ``gemini:`` and
``anthropic:``. `docs/MODELS.md` compares them.

`router._build` finds one by importing the module paths listed for it in
`router._FACTORIES` — both ``app.llm.<name>_provider`` and
``app.llm.providers.<name>_provider`` are tried, first one that imports wins —
and takes the first of ``PROVIDER``, ``provider()`` or the class named in that
same entry. Nothing is imported until somebody asks for that provider, so a
process configured for OpenAI alone never loads the Anthropic SDK and never
needs a key for it.

Deliberately empty of code. Importing this package must not build a client, read
an API key, or pull in an SDK.
"""
