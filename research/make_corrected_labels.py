"""Build annotations_corrected/ = Diss/annotations with the label-audit fixes applied.

Label policy (decided with the author): PARAGRAPH capitals only. Large chapter-opening
initials (one per book) are not targets. Every one of the 280 pages was checked by eye
(outputs/page_sheets/*.jpg); the fixes below are the only disagreements found.
"""
import shutil
import xml.etree.ElementTree as ET

from common import ORIGINAL_ANN_DIR as EUCLID_ANN_DIR, HERE

FIXES = {
    # page: (boxes to add, boxes to remove)  - pixel coords of jpegs_cropped
    "32811_150": ([(1545, 910, 1718, 1074), (1548, 1420, 1720, 1598)], []),  # two small 'S' never labelled
    "32811_199": ([], [(159, 2503, 745, 3181)]),  # Book XI chapter initial - not a paragraph capital
}

out = HERE / "annotations_corrected"
out.mkdir(exist_ok=True)
for f in EUCLID_ANN_DIR.glob("*.xml"):
    shutil.copy(f, out / f.name)

for pid, (add, remove) in FIXES.items():
    tree = ET.parse(out / f"{pid}.xml"); root = tree.getroot()
    for obj in list(root.findall("object")):
        bb = [int(float(obj.find(f"bndbox/{k}").text)) for k in ("xmin", "ymin", "xmax", "ymax")]
        if tuple(bb) in remove:
            root.remove(obj)
    for b in add:
        o = ET.SubElement(root, "object"); ET.SubElement(o, "name").text = "object"
        bb = ET.SubElement(o, "bndbox")
        for k, v in zip(("xmin", "ymin", "xmax", "ymax"), b):
            ET.SubElement(bb, k).text = str(v)
    tree.write(out / f"{pid}.xml", encoding="utf-8", xml_declaration=True)
print("wrote", out)
