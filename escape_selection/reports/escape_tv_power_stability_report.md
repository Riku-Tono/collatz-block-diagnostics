# ESCAPE-conditioned TV power stability check

Fixed throughout: `h=2`, `depth=4`, word alphabet `1 / 2 / 3+`. Each power uses the complete finite layer. The ESCAPE iid reference uses the same matched first-passage sampler as the previous check.

| power | ESCAPE samples | TV_all | TV_escape mean | TV_escape range (5 iid runs) | TV_escape > TV_all in every run |
|---:|---:|---:|---:|---:|:---:|
| 25 | 328,301 | 0.000009060 | 0.003733770 | 0.003384409–0.004200860 | yes |
| 26 | 651,706 | 0.000004768 | 0.004884531 | 0.003822402–0.005456825 | yes |
| 27 | 1,308,571 | 0.000002265 | 0.003710716 | 0.003222819–0.004064692 | yes |
| 28 | 2,616,918 | 0.000001132 | 0.003758958 | 0.003336415–0.004132020 | yes |

The range is only the minimum–maximum across matched-iid runs with seeds `20260625..20260629`. Each run used `500,000` samples. No new classification or additional statistic was introduced.

## Direction-only judgment

`TV_escape > TV_all` is stable for every tested power and every iid repeat.

**ESCAPE条件付けと差の増幅には関連がある。原因・機構は未同定。**

This closes the present discrepancy check at the requested descriptive level.

## Source files

- `C:\Users\yauki\Documents\Codex\2026-06-25\new-chat\work\collatz_escape_word_deficit.py`
- status caches: `work\status_cache\odd_only_status_p25.bin`, `C:\Users\yauki\Documents\Codex\2026-06-25\new-chat-2\work\status_cache\odd_only_status_p26.bin`, `work\status_cache\odd_only_status_p27.bin`, `C:\Users\yauki\Documents\Codex\2026-06-25\new-chat-2\work\status_cache\odd_only_status_p28.bin`
- `C:\Users\yauki\Documents\design\Collatz\log\一時README\Collatz-finite-block_README.md`

## Command

```powershell
python work/check_escape_tv_power_stability.py --powers 25 26 27 28 --h 2 --depth 4 --iid-samples 500000 --iid-repeats 5 --seed 20260625 --cache-dir work/status_cache --out-dir outputs
```
