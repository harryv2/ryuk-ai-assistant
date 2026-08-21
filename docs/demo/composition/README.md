# Demo video composition

The [HyperFrames](https://github.com/heygen-com/hyperframes) source for
`../demo.mp4`. To re-render:

```bash
mkdir assets && cp ../\*.png ../../gsuite-screenshots/g\*.png assets/
npm run check
npm run render
```

Scene timings and captions live in `index.html` — one `.clip` per scene, a
single paused GSAP timeline, deterministic output. `../VIDEO_SCRIPT.md` holds
the matching narration if a voiced HeyGen version is wanted later.
