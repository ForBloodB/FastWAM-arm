"""Exact physical-size A4 ChArUco target; detects the rendered PDF in QA."""
from pathlib import Path
import json
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm

out=Path(__file__).resolve().parents[1]/'calibration'
out.mkdir(exist_ok=True)
if not (out/'charuco_5x7_24mm.png').exists():
    import cv2
    dictionary=cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    board=cv2.aruco.CharucoBoard((5,7),.024,.018,dictionary)
    img=board.generateImage((1000,1400),marginSize=0,borderBits=1)
    cv2.imwrite(str(out/'charuco_5x7_24mm.png'),img)
metadata={'type':'ChArUco','dictionary':'DICT_4X4_50','squares_x':5,'squares_y':7,
          'square_length_m':.024,'marker_length_m':.018,'outer_size_mm':[120,168],
          'page_mm':[210,297],'print_scale':'100% actual size; no fit-to-page',
          'origin':'top-left of board, x right, y down in printed view'}
(out/'board.json').write_text(json.dumps(metadata,indent=2))
path=out/'ChArUco_A4_5x7_24mm.pdf'
c=canvas.Canvas(str(path),pagesize=A4,pageCompression=1)
c.setTitle('ReFlex Arm ChArUco calibration target - print at 100 percent')
c.setFont('Helvetica-Bold',16);c.drawString(20*mm,278*mm,'ReFlex Arm / ChArUco Calibration')
c.setFont('Helvetica',10)
for i,line in enumerate(['A4 portrait | PRINT ACTUAL SIZE / 100% | Do not fit to page',
                         '5 x 7 squares | square: 24 mm | marker: 18 mm | DICT_4X4_50',
                         'Black/white board: 120 x 168 mm. Keep the white margin.']):
    c.drawString(20*mm,(267-i*6)*mm,line)
c.drawImage(str(out/'charuco_5x7_24mm.png'),45*mm,73*mm,width=120*mm,height=168*mm)
c.setFont('Helvetica-Bold',9)
c.drawString(45*mm,245*mm,'TOP LEFT ORIGIN    +X -->')
c.setLineWidth(.5)
c.line(55*mm,53*mm,155*mm,53*mm)
for x in range(55,156,10):
    c.line(x*mm,51*mm,x*mm,55*mm)
c.drawCentredString(105*mm,46*mm,'This line must measure exactly 100 mm after printing.')
c.setFont('Helvetica',9)
c.drawString(20*mm,34*mm,'Check one square = 24 mm; outer rectangle = 120 x 168 mm.')
c.drawString(20*mm,28*mm,'Mount flat on rigid backing. Avoid glossy lamination, wrinkles and bending.')
c.drawString(20*mm,22*mm,'Use the full PDF page. Screen display is not a physical calibration target.')
c.showPage();c.save();print(path)
