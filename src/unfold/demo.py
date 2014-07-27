"""
Demo for seismic image unfolding.
"""

from common import *

# Dimensions for synthetic image.
n1,n2,n3 = 101,102,103
s1,s2,s3 = Sampling(n1),Sampling(n2),Sampling(n3)

# Names and descriptions of image files used below.
gxfile = "gx" # input image
u1file = "u1" # normal vector (1st component)
u2file = "u2" # normal vector (2nd component)
u3file = "u3" # normal vector (3rd component)
epfile = "ep" # eigenvalue-derived planarity
r1file = "r1" # unfolding shifts (1st component)
r2file = "r2" # unfolding shifts (2nd component)
r3file = "r3" # unfolding shifts (3rd component)
hxfile = "hx" # unfolded image 

# Directory for image files.
seismicDir = "./dat/"

# Directory for saved png images.
pngDir = None
#pngDir = "./png/"

def main(args):
  goFakeData()
  goNormalVectors()
  goUnfolding()

def goFakeData():
  sequence = 'OA' # 1 episode of folding, followed by one episode of faulting
  impedance = False # if True, data = impedance model
  wavelet = True # if False, no wavelet will be used
  noise = 0.5 # (rms noise)/(rms signal) ratio
  gx,_,_ = FakeData.seismicAndSlopes3d2014A(
      sequence,0,False,False,impedance,wavelet,noise)
  writeImage(gxfile,gx)
  print "gx min =",min(gx)," max =",max(gx)
  gmin,gmax,gmap = -3.0,3.0,ColorMap.GRAY
  if impedance:
    gmin,gmax,gmap = 0.0,1.4,ColorMap.JET
  plot3(gx,cmin=gmin,cmax=gmax,cmap=gmap,clab="Amplitude",png="gx")

def goNormalVectors():
  u1 = zerofloat(n1,n2,n3)
  u2 = zerofloat(n1,n2,n3)
  u3 = zerofloat(n1,n2,n3)
  ep = zerofloat(n1,n2,n3)
  gx = readImage(gxfile)
  lof = LocalOrientFilter(8.0,2.0)
  #lof = LocalOrientFilter(4.0,1.0)
  lof.applyForNormalPlanar(gx,u1,u2,u3,ep)
  writeImage(u1file,u1)
  writeImage(u2file,u2)
  writeImage(u3file,u3)
  writeImage(epfile,ep)

def goUnfolding():
  hx = zerofloat(n1,n2,n3)
  gx = readImage(gxfile)
  u1 = readImage(u1file)
  u2 = readImage(u2file)
  u3 = readImage(u3file)
  ep = readImage(epfile)
  pow(ep,8.0,ep)
  p = array(u1,u2,u3,ep)
  flattener = FlattenerRT(6.0,6.0)
  r = flattener.findShifts(p)
  flattener.applyShifts(r,gx,hx)
  writeImage(r1file,r[0])
  writeImage(r2file,r[1])
  writeImage(r3file,r[2])
  writeImage(hxfile,hx)
  hmin,hmax,hmap = -3.0,3.0,ColorMap.GRAY
  plot3(hx,cmin=hmin,cmax=hmax,cmap=hmap,clab="Amplitude",png="hx")

def array(x1,x2,x3=None,x4=None):
  if x3 and x4:
    return jarray.array([x1,x2,x3,x4],Class.forName('[[[F'))
  elif x3:
    return jarray.array([x1,x2,x3],Class.forName('[[[F'))
  else:
    return jarray.array([x1,x2],Class.forName('[[[F'))

#############################################################################
# graphics

def addColorBar(frame,clab=None,cint=None):
  cbar = ColorBar(clab)
  if cint:
    cbar.setInterval(cint)
  cbar.setFont(Font("Arial",Font.PLAIN,32)) # size by experimenting
  cbar.setWidthMinimum
  cbar.setBackground(Color.WHITE)
  frame.add(cbar,BorderLayout.EAST)
  return cbar

def plot3(f,cmin=None,cmax=None,cmap=None,clab=None,cint=None,png=None):
  n1,n2,n3 = len(f[0][0]),len(f[0]),len(f)
  sf = SimpleFrame(AxesOrientation.XRIGHT_YOUT_ZDOWN)
  cbar = None
  ipg = sf.addImagePanels(s1,s2,s3,f)
  if cmap!=None:
    ipg.setColorModel(cmap)
  if cmin!=None and cmax!=None:
    ipg.setClips(cmin,cmax)
  else:
    ipg.setClips(-3.0,3.0)
  if clab:
    cbar = addColorBar(sf,clab,cint)
    ipg.addColorMapListener(cbar)
  if cbar:
    cbar.setWidthMinimum(120)
  ipg.setSlices(95,5,51)
  #ipg.setSlices(95,5,95)
  if cbar:
    sf.setSize(837,700)
  else:
    sf.setSize(700,700)
  vc = sf.getViewCanvas()
  vc.setBackground(Color.WHITE)
  radius = 0.5*sqrt(n1*n1+n2*n2+n3*n3)
  ov = sf.getOrbitView()
  ov.setWorldSphere(BoundingSphere(0.5*n1,0.5*n2,0.5*n3,radius))
  ov.setAzimuthAndElevation(-55.0,25.0)
  ov.setTranslate(Vector3(0.0241,0.0517,0.0103))
  ov.setScale(1.2)
  sf.setVisible(True)
  if png and pngDir:
    sf.paintToFile(pngDir+png+".png")
    if cbar:
      cbar.paintToPng(137,1,pngDir+png+"cbar.png")

#############################################################################
# read/write images

def readImage(name):
  """ 
  Reads an image from a file with specified name.
  name: base name of image file; e.g., "tpsz"
  """
  fileName = seismicDir+name+".dat"
  image = zerofloat(n1,n2,n3)
  ais = ArrayInputStream(fileName)
  ais.readFloats(image)
  ais.close()
  return image

def writeImage(name,image):
  """ 
  Writes an image to a file with specified name.
  name: base name of image file; e.g., "tpgp"
  image: the image
  """
  fileName = seismicDir+name+".dat"
  aos = ArrayOutputStream(fileName)
  aos.writeFloats(image)
  aos.close()
  return image

#############################################################################
run(main)
