import io,re
from pathlib import Path
import cv2,joblib,numpy as np,pandas as pd,streamlit as st
from PIL import Image
from skimage.feature import hog
from openpyxl import load_workbook
from openpyxl.utils.cell import coordinate_from_string,column_index_from_string,get_column_letter
B=Path(__file__).parent;MODELS=joblib.load(B/'digit_models.joblib');SLOTS=[(0,39),(35,74),(82,124)]
st.set_page_config(page_title='測定値OCR',page_icon='🔢',layout='wide');st.title('🔢 測定値を自動認識してExcelへ');st.caption('加工前写真専用。表示はXX.X、範囲は20.0～99.9です。')
def timekey(n):
 m=re.search(r'_(\d+(?:\.\d+)?)s(?:\.[^.]+)?$',n,re.I);return (0,float(m.group(1))) if m else (1,n.lower())
def align(rgb):
 h,w=rgb.shape[:2];cx=round(w*.505);cy=round(h*.544);size=max(20,round(min(h,w)*.072));u=size/30;b=(max(0,int(cx-48*u)),max(0,int(cy-33*u)),min(w,int(cx-1*u)),min(h,int(cy-3*u)))
 r=rgb[b[1]:b[3],b[0]:b[2]]
 if r.size==0:return None,b
 return cv2.resize(cv2.cvtColor(r,cv2.COLOR_RGB2GRAY),(126,72)),b
def feature(q):
 q=cv2.resize(q,(32,48));q=cv2.createCLAHE(2,(4,4)).apply(q);return hog(q,orientations=9,pixels_per_cell=(8,8),cells_per_block=(2,2),block_norm='L2-Hys')
def recognize(rgb):
 a,b=align(rgb)
 if a is None:return '',0,b,[],None
 out='';details=[];cs=[]
 for pos,(x1,x2) in enumerate(SLOTS):
  f=feature(a[2:70,x1:x2]);m=MODELS[pos];pr=m.predict_proba([f])[0];order=np.argsort(pr)[::-1];d=str(m.classes_[order[0]]);out+=d;cs.append(float(pr[order[0]]));details.append([(str(m.classes_[j]),float(pr[j])) for j in order[:4]])
 v=float(out[:2]+'.'+out[2])
 if not 20<=v<=99.9:return out[:2]+'.'+out[2],0,b,details,a
 return out[:2]+'.'+out[2],float(np.mean(cs)),b,details,a
def cellok(x):return bool(re.fullmatch(r'[A-Za-z]{1,3}[1-9][0-9]*',x.strip()))
excel=st.file_uploader('1. 入力先Excel',type=['xlsx']);fs=sorted(st.file_uploader('2. 加工前写真を選択',type=['png','jpg','jpeg','webp'],accept_multiple_files=True) or [],key=lambda f:timekey(f.name));rows=[]
if fs:st.info('時間順：'+' → '.join(f.name for f in fs))
for i,f in enumerate(fs):
 try:rgb=np.array(Image.open(io.BytesIO(f.getvalue())).convert('RGB'))
 except Exception as e:st.error(f'{f.name}を開けません：{e}');continue
 v,c,b,d,a=recognize(rgb);mark=rgb.copy();cv2.rectangle(mark,(b[0],b[1]),(b[2],b[3]),(0,255,0),2);l,r=st.columns([1,2]);l.image(mark,caption=f.name,width='stretch');v=r.text_input('認識結果',v,key=f'{i}_{f.name}');r.write(f'平均確率 {c:.2f}')
 if c<.55:r.warning('確率が低いため確認してください。')
 with r.expander('候補を見る'):
  if a is not None:r.image(a,caption='認識に使った数値部分',width='stretch')
  for k,z in enumerate(d):r.write(f'{k+1}桁目：'+', '.join(f'{x}({p:.2f})' for x,p in z))
 rows.append({'filename':f.name,'value':v,'confidence':round(c,3)})
if rows:
 st.dataframe(pd.DataFrame(rows),width='stretch')
 if excel:
  wb=load_workbook(io.BytesIO(excel.getvalue()));sn=st.selectbox('入力シート',wb.sheetnames);start=st.text_input('開始セル','C4')
  if cellok(start):
   letters,r0=coordinate_from_string(start.upper());col=column_index_from_string(letters);ws=wb[sn];bad=[]
   for j,x in enumerate(rows):
    try:cell=ws.cell(r0+j,col,float(x['value']));cell.number_format='0.0'
    except:bad.append(x['filename'])
   if bad:st.error('数値にできない画像：'+'、'.join(bad))
   else:
    out=io.BytesIO();wb.save(out);end=f'{get_column_letter(col)}{r0+len(rows)-1}';st.success(f'{start.upper()}:{end}へ入力しました');st.download_button('入力済みExcelをダウンロード',out.getvalue(),f'{Path(excel.name).stem}_入力済み.xlsx')
