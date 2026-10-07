import io, re, csv, zipfile, hashlib
from pathlib import Path
import cv2, numpy as np, pandas as pd, streamlit as st
from PIL import Image
from rapidocr import RapidOCR
from openpyxl import load_workbook
from openpyxl.utils.cell import coordinate_from_string, column_index_from_string, get_column_letter

st.set_page_config(page_title='測定値OCR 60枚対応', page_icon='🔢', layout='wide')
st.title('🔢 測定値OCR 70枚対応版')
st.caption('高速OCRを先に実行し、未検出画像だけ追加OCRします。同じ画像は再処理しません。')

@st.cache_resource
def get_ocr(): return RapidOCR()
OCR=get_ocr()
if 'results' not in st.session_state: st.session_state.results={}
if 'corrections' not in st.session_state: st.session_state.corrections={}

def timekey(n):
 m=re.search(r'_(\d+(?:\.\d+)?)s(?:\.[^.]+)?$',n,re.I); return (0,float(m.group(1))) if m else (1,n.lower())
def valid(v): return bool(re.fullmatch(r'\d{2}\.\d',str(v).strip())) and 20<=float(v)<=99.9
def clean(t): return str(t).translate(str.maketrans({'O':'0','o':'0','I':'1','l':'1','|':'1','S':'5','s':'5','B':'8','，':'.','。':'.',',':'.','・':'.'}))
def extract(text,score,tag):
 t=clean(text); out=[]
 for m in re.finditer(r'(\d{2})\s*\.\s*(\d)',t):
  v=float(m.group(1)+'.'+m.group(2));
  if 20<=v<=99.9: out.append((v,score+.20,tag,text))
 d=''.join(re.findall(r'\d',t))
 if len(d)>=3:
  for i in range(len(d)-2):
   v=float(d[i:i+2]+'.'+d[i+2]);
   if 20<=v<=99.9: out.append((v,score+(0.04 if i==0 else 0),tag,text))
 return out

def ocr_once(tag,img,scale=3.0):
 r=OCR(cv2.resize(img,None,fx=scale,fy=scale,interpolation=cv2.INTER_CUBIC)); rows=[]; cans=[]
 if r is not None and r.txts is not None:
  rows=[(str(t),float(s)) for t,s in zip(r.txts,r.scores)]
  for t,s in rows: cans.extend(extract(t,s,tag))
 return rows,cans

def rank(cans):
 if not cans:return '未検出',0.0,[]
 g={}
 for v,s,tag,t in cans:g.setdefault(v,[]).append((s,tag,t))
 ranked=[]
 for v,e in g.items():
  methods=len(set(x[1] for x in e)); ranked.append((max(x[0] for x in e)+.10*min(methods-1,3),v,e))
 ranked.sort(reverse=True);return f'{ranked[0][1]:.1f}',min(1.0,ranked[0][0]),ranked[:5]

def recognize_bytes(data):
 rgb=np.array(Image.open(io.BytesIO(data)).convert('RGB'));bgr=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR);h,w=bgr.shape[:2]
 center=bgr[int(h*.34):int(h*.70),int(w*.16):int(w*.82)]
 logs=[];cans=[]
 # Fast stage: only two calls.
 for tag,img in [('中央',center),('画像全体',bgr)]:
  rows,c=ocr_once(tag,img,3.0 if tag=='中央' else 2.3);logs.append((tag,rows));cans+=c
 value,conf,ranked=rank(cans)
 # Difficult images only: up to three extra calls.
 if value=='未検出' or conf<.72:
  gray=cv2.cvtColor(center,cv2.COLOR_BGR2GRAY);clahe=cv2.createCLAHE(2,(8,8)).apply(gray);adaptive=cv2.adaptiveThreshold(clahe,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,31,7)
  for tag,img in [('コントラスト',clahe),('二値化',adaptive),('反転',255-adaptive)]:
   rows,c=ocr_once(tag,img,3.5);logs.append((tag,rows));cans+=c
  value,conf,ranked=rank(cans)
 return {'value':value,'confidence':conf,'logs':logs,'ranking':[(s,v) for s,v,_ in ranked]}

def corrections_zip():
 out=io.BytesIO();rows=[]
 with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:
  for fn,r in st.session_state.corrections.items():
   rows.append({'filename':fn,'ocr_result':r['ocr'],'correct_value':r['correct'],'confidence':r['confidence']});z.writestr('images/'+fn,r['bytes'])
  txt=io.StringIO();w=csv.DictWriter(txt,fieldnames=['filename','ocr_result','correct_value','confidence']);w.writeheader();w.writerows(rows);z.writestr('corrections.csv',txt.getvalue().encode('utf-8-sig'))
 return out.getvalue()

excel=st.file_uploader('1. 入力先Excel',type=['xlsx'])
files=sorted(st.file_uploader('2. 加工前写真を選択（最大70枚）',type=['png','jpg','jpeg','webp'],accept_multiple_files=True) or [],key=lambda f:timekey(f.name))
if len(files)>70:st.error('一度に選択できるのは70枚までです。');st.stop()
if files:st.info(f'{len(files)}枚：'+' → '.join(f.name for f in files[:8])+(' …' if len(files)>8 else ''))
if files and st.button('選択した写真をまとめて認識',type='primary'):
 bar=st.progress(0);status=st.empty()
 for i,f in enumerate(files):
  data=f.getvalue();key=hashlib.sha256(data).hexdigest()
  if key not in st.session_state.results:
   status.write(f'認識中 {i+1}/{len(files)}：{f.name}')
   try:st.session_state.results[key]=recognize_bytes(data)
   except Exception as e:st.session_state.results[key]={'value':'未検出','confidence':0.0,'logs':[('エラー',[(str(e),0.0)])],'ranking':[]}
  bar.progress((i+1)/len(files))
 status.success('認識が完了しました。');bar.empty()

rows=[]
for i,f in enumerate(files):
 data=f.getvalue();key=hashlib.sha256(data).hexdigest();res=st.session_state.results.get(key)
 if not res:continue
 with st.expander(f'{i+1}. {f.name}  |  {res["value"]}  |  確度 {res["confidence"]:.2f}',expanded=res['value']=='未検出' or res['confidence']<.70):
  l,r=st.columns([1,2]);l.image(data,width='stretch')
  value=r.text_input('認識結果（必要なら修正）','' if res['value']=='未検出' else res['value'],key='v_'+key,placeholder='例：45.1')
  if r.button('修正データとして保存',key='s_'+key):
   if not valid(value):r.error('20.0～99.9のXX.X形式で入力してください。')
   elif value==res['value']:r.info('OCR結果と同じため保存しません。')
   else:st.session_state.corrections[f.name]={'ocr':res['value'],'correct':value,'confidence':round(res['confidence'],4),'bytes':data};r.success('保存しました。')
  for tag,items in res['logs']:
   if items:r.write(tag+'：'+'、'.join(f'{t}({s:.2f})' for t,s in items))
  rows.append({'filename':f.name,'value':value,'confidence':round(res['confidence'],3)})

if st.session_state.corrections:
 st.download_button('修正データ一式をダウンロード',corrections_zip(),'ocr_corrections.zip','application/zip')
if rows:
 st.dataframe(pd.DataFrame(rows),width='stretch')
 if excel:
  default_output_name=f'{Path(excel.name).stem}_入力済み.xlsx'
  output_name=st.text_input('ダウンロードするExcelのファイル名',default_output_name)
  output_name=output_name.strip() or default_output_name
  if not output_name.lower().endswith('.xlsx'):output_name += '.xlsx'
  output_name=re.sub(r'[\\/:*?"<>|]+','_',output_name)
  wb=load_workbook(io.BytesIO(excel.getvalue()));sn=st.selectbox('入力シート',wb.sheetnames);start=st.text_input('開始セル','C4')
  if re.fullmatch(r'[A-Za-z]{1,3}[1-9][0-9]*',start.strip()):
   letters,r0=coordinate_from_string(start.upper());col=column_index_from_string(letters);ws=wb[sn];bad=[]
   for j,row in enumerate(rows):
    try:v=float(row['value']);assert 20<=v<=99.9;cell=ws.cell(r0+j,col,v);cell.number_format='0.0'
    except:bad.append(row['filename'])
   if bad:st.error('未検出または無効な値：'+'、'.join(bad))
   else:
    out=io.BytesIO();wb.save(out);end=f'{get_column_letter(col)}{r0+len(rows)-1}';st.success(f'{start.upper()}:{end}へ入力しました');st.download_button('入力済みExcelをダウンロード',out.getvalue(),output_name,mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
