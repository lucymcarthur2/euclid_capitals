# Initials Studio

Find the decorated woodcut initials in scanned early printed books, check them, and group them by the
**physical woodblock** they were printed from, across books and printers. If two printers' books show
impressions of the same block, the block was lent, sold or inherited.

Built for the question of whether Venetian printers of the 1480s, Erhard Ratdolt and the men around him,
shared their woodblocks. The bundled examples already show Ratdolt's blocks in books printed in Venice
in 1488 by Thomas de Blavis and by Santritter & de Sanctis, two years after Ratdolt had left for Augsburg.

Everything runs on your own computer. Nothing is uploaded anywhere.

## Start it

**Windows, no installation:** unzip `Initials Studio (Windows).zip` anywhere, open the folder, and double-click
**`Initials Studio.exe`**. Windows may warn that the app is from an unknown publisher, because it isn't
code-signed; choose *More info → Run anyway*. A small black window opens (close it to stop the app), and the
app opens in your web browser. Nothing else needs installing.

**From source (Windows, macOS, Linux):** with Python 3.10+ installed, double-click
`Start Initials Studio (Windows).bat` or run `./start.sh`.

The first start installs what it needs into a private `.venv` folder. That takes about 300 MB, 5-10 minutes,
and an internet connection. Python 3.10 or newer must be installed ([python.org](https://www.python.org/downloads/);
on Windows tick *Add python.exe to PATH*). Later starts take a few seconds. The app opens in your web browser
at `http://127.0.0.1:8800`. Close the black window to stop it.

No GPU is needed. Finding capitals takes about 0.3-0.5 s per page on an ordinary laptop.

## Using it

1. **Library.** Drop in PDFs or page images (JPG, PNG, TIFF). Each PDF becomes a book, or all images become one.
   Give the printer: the Woodblocks view uses it to spot blocks *shared between printers*.
2. **Review.** Every capital the model found, with **flags** on anything that looks doubtful:

   | Flag | Usually means |
   |---|---|
   | Low confidence | the model wasn't sure; often damaged, faint or unusual initials |
   | Unusual size for this book | a chapter initial, a picture, or a box around two letters |
   | Not square like a woodcut initial | part of a picture or a run of text |
   | No clear woodcut frame | a printed (non-woodcut) letter, a diagram, show-through |
   | Faint | show-through from the other side of the leaf, or a damaged print |
   | Overlaps another box / Touches the page edge | a duplicate or cut-off detection |
   | Unclear block match | close to a woodblock but not convincingly the same: worn block or recut copy? |
   | Mirror image of another block | a reversed recut copy (a block can never print mirrored) |

   Click ✓ Keep or ✕ Out. Use *Pages* to see whole pages. Click a box to keep or reject it, and drag to add a
   capital the model missed. *Below the cut-off* lists near misses worth a glance.
3. **Woodblocks.** Press **Update grouping**. Every capital is compared with every other, in all four
   rotations, and grouped by block. Each card is one block, with its impressions colour-coded by book.
   Initials printed upside down or sideways are turned back and marked ↻. Select impressions and *Move here*,
   or *Make a new woodblock*, to correct the grouping. Double-click to compare two impressions large.
   Adding a book later only scores the new comparisons, and keeps your corrections.
4. **Export.** A CSV with every capital (book, page, box, confidence, woodblock, orientation, match score),
   or a zip with the crops sorted into one folder per woodblock.

**Settings** change the confidence cut-off and flag sensitivity, which take effect instantly, and the grouping
thresholds, which apply at the next *Update grouping*.

## How it works

The app's **About the model** tab has the full write-up: the training data and label policy, the detector
and how it was trained, the testing method, results, comparison with the dissertation model, flags, the
woodblock matching and its calibration, assumptions and limitations. A summary:

**Finding capitals:** a Faster R-CNN detector (MobileNetV3-Large FPN backbone, COCO-pretrained), fine-tuned on
real pages of Euclid's *Elements* (Ratdolt, Venice 1482). Three more Ratdolt-era books were added to
training (Ptolemy, Albumasar, Boethius' *Arithmetica*) so their woodcut pictures teach the model what is
*not* a capital. Label policy: paragraph initials, not large chapter initials.

| Test | Capitals | Found | False alarms |
|---|---|---|---|
| Euclid, 5-fold cross-validation (cut-off chosen per fold) | 510 | 507 | 0 |
| Compilatio (never seen in training), cut-off 0.5 | 109 | 106 | 3 |
| Artes orandi (never seen), cut-off 0.5 | 93 | 93 | 8 (one chessboard page) |
| Judgement of the Stars (never seen; photos of an open book) | 362 | 362 | 2 |

The default cut-off for new books is 0.5. 0.94 is a strict, Euclid-tuned setting that misses more on new books.

**Grouping:** each capital is cropped to its black frame and squared. Two capitals are aligned with
OpenCV's ECC image registration, which allows a small shift, scale and skew, and their ink patterns are
correlated. A quick 8-orientation check of tiny thumbnails picks each capital's closest candidates first,
so only about 10% of pairs need the full comparison.

On the bundled books the scores fall into clear ranges:

| Pair | Typical score |
|---|---|
| Same block, same book | ~0.92 |
| Same block, two Ratdolt books | ~0.87 |
| Same block, Ratdolt and another printer | ~0.78 |
| Different designs, even of the same letter | < 0.5 |

Capitals are grouped by **average** linkage at 0.70, so a smudged look-alike cannot chain two blocks together.
The 0.5-0.7 band is flagged *unclear*.

The cross-printer scores are lower partly because the scans differ and partly because the blocks wore down.
That is why the *unclear* matches are worth your eye.

## The bundled examples

| Book | Printer | In the app |
|---|---|---|
| Euclid, *Elementa* (1482) | Erhard Ratdolt | capitals only (510) |
| Leopold of Austria, *Compilatio* (1489) | Erhard Ratdolt | capitals only |
| *Artes orandi* | Erhard Ratdolt | capitals only |
| Ptolemy | Erhard Ratdolt | capitals only |
| *Judgement of the Stars* | Erhard Ratdolt | capitals only |
| Jandun, *Quaestiones* (Venice 1488) | Santritter & de Sanctis | capitals only (BSB Munich, Public Domain Mark) |
| Hyginus, *Poeticon astronomicon* (Venice 1488) | Thomas de Blavis | **full pages** (Université de Liège, public domain) |
| Aquinas, *De esse et essentia* (Venice 1488) | Santritter & de Sanctis | **full pages** (BSB Munich, Public Domain Mark) |

"Capitals only" books contain just the cropped initials, not the page scans. Before sharing the app publicly,
check that the libraries holding the Ratdolt scans allow redistribution of crops. Delete an example book in
the app if you don't want it.

## Limits worth knowing

- The detector is trained on Ratdolt-style **white-on-black vine initials**. Hand-painted initials and
  printed (type) letters are not woodcuts and are not found. Very differently styled woodcuts, such as
  later or northern European ones, may need more training pages.
- Heavily hand-coloured copies lower the confidence. Use a lower cut-off and check the flagged ones.
- "Same block" is a visual judgement from scans. Confirm important matches by comparing shared damage,
  such as cracks and broken lines, in the large comparison view.

## Files

- `workspace/`: your books, decisions and grouping. Back it up to keep your work; delete it to start fresh.
- `models/`: the detector weights and their description.
- `examples/`: the bundled examples, copied into the workspace on first start.
- `initials_studio/`: the application (Python + a single-page web interface).
