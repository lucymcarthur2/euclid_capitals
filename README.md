# Euclid's capitals: finding and matching woodcut initials in early printed books

Code accompanying *Investigating early printing practices through computational analysis of Euclid's
Elements* (McArthur, Zhou, Tuckett, Scanlon & Gibson, submitted to *Heritage*), and its follow-on tool.

Early printers decorated their books with woodcut initials and reused the same blocks again and again,
within a book, across books, and sometimes between printing houses. Finding every initial in a book and
grouping the impressions by the physical woodblock that printed them gives systematic evidence for how
workshops managed their stock and whether blocks moved between printers.

| Folder | What it is |
|---|---|
| [`dissertation/`](dissertation/) | The code from the MSc dissertation that the paper describes: synthetic training pages, a Faster R-CNN detector (ResNet-50 + FPN), extraction, and ORB-based grouping by woodblock. |
| [`initials_studio/`](initials_studio/) | **Initials Studio**: a desktop app that finds, checks and groups woodcut initials in any scanned book (PDF or images). |
| [`research/`](research/) | The follow-on research code behind Initials Studio: an improved detector trained on real labelled pages, cross-book evaluation, and the woodblock matcher. Includes the label files. |

## Initials Studio

**Windows, no installation:** download `Initials Studio (Windows).zip` from the
[Releases](../../releases) page, unzip it, and double-click `Initials Studio.exe`.

**From source (Windows, macOS, Linux):** with Python 3.10+, download the model weights
(`capitals_mbv3_v2.pth`) from [Releases](../../releases) into `initials_studio/models/`, then run
`initials_studio/start.sh` or `Start Initials Studio (Windows).bat`.

The app finds the initials, flags doubtful detections for checking, groups every impression by woodblock
across all your books (including blocks printed turned), and exports the results as CSV and image folders.
See [`initials_studio/README.md`](initials_studio/README.md).

## Results in brief

The improved detector, trained on real pages:

| Book | Capitals | Result |
|---|---|---|
| Euclid (5-fold cross-validation) | 510 | 507 found, 0 false alarms |
| Compilatio (unseen) | 109 | F1 0.97 |
| Artes orandi (unseen) | 93 | F1 0.96–0.995, depending on cut-off |

The dissertation model reaches F1 0.986 on Euclid. Details are in [`research/results/`](research/results/)
and in the app's *About the model* page.

## Data

The scans are **not** included. They belong to the holding libraries:

- **UCL Special Collections:** *Elements*, 1482.
- **Library of Congress:** *Compilatio*, 1489; *Artes orandi*, 1485; the gold-leaf *Elements*.
- **Bayerische Staatsbibliothek** and **Université de Liège:** the 1488 Venetian books.

Bounding-box labels are in `research/labels/`. The scripts expect scans in a `data/` folder next to them.

## Licence

MIT (see [LICENSE](LICENSE)). If you use this code, please cite the paper.
