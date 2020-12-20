class TextProcess:
  def __init__(self):
    self.nepali_letters = ['क', 'ख', 'ग', 'घ','ङ','च', 'छ', 'ज', 'झ', 'ञ', 'ट', 'ठ', 'ड', 'ढ', 'ण', 'त', 'थ', 'द','ध','न','प','फ','ब','भ','म','य', 'र', 'ल', 'व','श','ष','स','ह','क्ष','त्र','त्ज्ञ','अ','आ','इ','ई','उ','ऊ','ए','ऐ','ओ','औ','अं','अः', 'ा','ि','ी','ु','ू','े','ै','ो','ौ', 'ं', 'ँ','्','ृ','ः','०', '१', '२', '३', '४', '५', '६', '७','८', '९','।',' ']
  
  def text_to_int(self, ip_str):
    indices = []
    for data in ip_str:
      indices.append(self.nepali_letters.index(data))
    return indices
  def int_to_text(self, seq):
    string = []
    for i in seq:
      string.append(self.nepali_letters[i])
      return ''.join(string)


def processLabels(labels):
  retVal = []
  for i in labels:
    s = ""
    for l in i:
      if (l=='\u200d' or l =='\u200c'):
        continue
      elif (l == 'ऋ' or l == 'ॠ'):
        s+= 'र' + 'ि'
      else:
        s+=l
    retVal.append(s)
  return retVal