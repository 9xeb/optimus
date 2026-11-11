# Vendored libraries

These files are committed to the repository and copied verbatim during deployment. They are not downloaded, regenerated, or bundled at deploy time. Do not edit them by hand.

| Library | Version | File | Upstream | SHA-256 | License |
|---|---|---|---|---|---|
| d3 | 7.9.2 | `d3.min.js` | https://cdn.jsdelivr.net/npm/d3@7.9.2/dist/d3.min.js | `912111098ea99503a6f4967361d2bbdc9dbf58cf9c678c8e27e5b02d1a07f0c1` | ISC |
| @dagrejs/dagre | 1.1.8 | `dagre.min.js` | https://cdn.jsdelivr.net/npm/@dagrejs/dagre@1.1.8/dist/dagre.min.js | `c35b8d6f410ce7bdd302fc00cad27331184e52b2e738609e55ae3bbb4fb4849f` | MIT (includes graphlib, MIT) |
| marked | 13.0.3 | `marked.min.js` | https://cdn.jsdelivr.net/npm/marked@13.0.3/marked.min.js | `5adea7d8ee41a700fccc14bb9d503104f0470cc17a84ad3e167d3f5251eae0da` | MIT |
| DOMPurify | 3.4.17 | `purify.min.js` | https://cdn.jsdelivr.net/npm/dompurify@3.4.17/dist/purify.min.js | `cd37898f9782adf54820c413191b5691803f0f137c799d6edcfd38523b4c2c31` | MPL-2.0 OR Apache-2.0 |

All four are loaded as classic scripts and expose the globals `d3`, `dagre`, `marked`, and `DOMPurify`.

## Upgrading

Replace the file and update its version and SHA-256 hash in the table above in the same change. Verify the new file still exposes the expected global and does not contain a `</script` sequence before using it inline anywhere. Do not add a CDN reference to `index.html` or `assets/`.

```sh
curl -fsSL 'https://cdn.jsdelivr.net/npm/<package>@<version>/<path>' -o 'vendor/<file>'
sha256sum 'vendor/<file>'
```
