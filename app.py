import io, re, csv, zipfile
from pathlib import Path
import cv2, numpy as np, pandas as pd, streamlit as st
from PIL import Image
from rapidocr import RapidOCR
from openpyxl import load_workbook
from openpyxl.utils.cell import coordinate_from_string, column_index_from_string, get_column_letter

st.set_page_config(page_title='測定値OCR', page_icon='🔢', layout='wide')
st.title('🔢 複数範囲・複数処理で「XX.X」を読み取り')
st.caption('加工前写真専用。未検出は空欄にせず「未検出」と表示し、修正画像をZIP保存できます。')

@st.cache_resource
def get_ocr(): return RapidOCR()
OCR = get_ocr()
if 'corrections' not in st.session_state: st.session_state.corrections = {}

def timekey(name):
    m=re.search(r'_(\d+(?:\.\d+)?)s(?:\.[^.]+)?$',name,re.I)
    return (0,float(m.group(1))) if m else (1,name.lower())

def valid_value(v):
    v=str(v).strip()
    return bool(re.fullmatch(r'\d{2}\.\d',v)) and 20<=float(v)<=99.9

def normalize_text(t):
    return str(t).translate(str.maketrans({'O':'0','o':'0','I':'1','l':'1','|':'1','S':'5','s':'5','B':'8','，':'.','。':'.',',':'.','・':'.'}))

def numeric_candidates(text,score,tag):
    t=normalize_text(text); out=[]
    for m in re.finditer(r'(\d{2})\s*\.\s*(\d)',t):
        v=float(m.group(1)+'.'+m.group(2))
        if 20<=v<=99.9: out.append((v,score+0.20,tag,text))
    digits=''.join(re.findall(r'\d',t))
    if len(digits)>=3:
        for i in range(len(digits)-2):
            v=float(digits[i:i+2]+'.'+digits[i+2])
            if 20<=v<=99.9: out.append((v,score+(0.04 if i==0 else 0),tag,text))
    return out

def join_spatial(result):
    """同じ行で近接するOCR断片を、座標の左から連結する。"""
    if result is None or result.txts is None or result.boxes is None: return []
    items=[]
    for box,text,score in zip(result.boxes,result.txts,result.scores):
        box=np.asarray(box); x1,x2=float(box[:,0].min()),float(box[:,0].max()); y1,y2=float(box[:,1].min()),float(box[:,1].max())
        items.append({'x1':x1,'x2':x2,'y':(y1+y2)/2,'h':max(1,y2-y1),'t':str(text),'s':float(score)})
    joined=[]
    for seed in items:
        line=[x for x in items if abs(x['y']-seed['y'])<=max(x['h'],seed['h'])*0.75]
        line=sorted(line,key=lambda x:x['x1'])
        chunks=[]
        for x in line:
            if not chunks or x['x1']-chunks[-1][-1]['x2']<=max(x['h'],chunks[-1][-1]['h'])*2.2: chunks.append((chunks.pop() if chunks else [])+[x])
            else: chunks.append([x])
        for chunk in chunks:
            if seed not in chunk: continue
            joined.append((''.join(x['t'] for x in chunk),float(np.mean([x['s'] for x in chunk]))))
    # de-duplicate
    seen=set(); out=[]
    for x in joined:
        if x[0] not in seen: seen.add(x[0]); out.append(x)
    return out

def preprocess_variants(crop):
    gray=cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY)
    clahe=cv2.createCLAHE(2.0,(8,8)).apply(gray)
    adaptive=cv2.adaptiveThreshold(clahe,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,31,7)
    otsu=cv2.threshold(clahe,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    return [('通常',crop),('グレー',gray),('コントラスト',clahe),('適応二値',adaptive),('反転',255-otsu)]

def read_one(tag,image,scale):
    up=cv2.resize(image,None,fx=scale,fy=scale,interpolation=cv2.INTER_CUBIC)
    result=OCR(up)
    rows=[]
    if result is not None and result.txts is not None:
        rows=[(str(t),float(s)) for t,s in zip(result.txts,result.scores)]
        rows += join_spatial(result)
    cands=[]
    for t,s in rows: cands += numeric_candidates(t,s,tag)
    return rows,cands

def recognize(rgb):
    bgr=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR); h,w=bgr.shape[:2]
    ranges=[
      ('中央・広',(.14,.82,.31,.72)),('中央',(.20,.76,.37,.68)),('中央・狭',(.28,.68,.42,.63)),
      ('少し上',(.18,.78,.30,.59)),('少し下',(.18,.78,.45,.75))]
    allc=[]; logs=[]; previews=[]
    # full image only once
    rows,c=read_one('全体・通常',bgr,2.5); logs.append(('全体・通常',rows)); allc+=c
    for rname,(xl,xr,yt,yb) in ranges:
        crop=bgr[int(h*yt):int(h*yb),int(w*xl):int(w*xr)]
        previews.append((rname,crop))
        for pname,img in preprocess_variants(crop):
            # Keep calls bounded: all five processes for central, key processes elsewhere
            if rname!='中央' and pname not in ('通常','コントラスト','適応二値'): continue
            tag=f'{rname}・{pname}'; rows,c=read_one(tag,img,3.5); logs.append((tag,rows)); allc+=c
    if not allc: return '未検出',0.0,logs,previews,[]
    grouped={}
    for v,s,tag,text in allc: grouped.setdefault(v,[]).append((s,tag,text))
    ranked=[]
    for v,ev in grouped.items():
        best=max(x[0] for x in ev); methods=len(set(x[1] for x in ev)); score=best+0.10*min(methods-1,4)
        ranked.append((score,v,ev))
    ranked.sort(reverse=True); best=ranked[0]
    return f'{best[1]:.1f}',min(1.0,best[0]),logs,previews,ranked[:8]

def correction_zip():
    out=io.BytesIO(); rows=[]
    with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:
        for fn,r in st.session_state.corrections.items():
            rows.append({'filename':fn,'ocr_result':r['ocr'],'correct_value':r['correct'],'confidence':r['confidence']})
            z.writestr('images/'+fn,r['bytes'])
        text=io.StringIO(); w=csv.DictWriter(text,fieldnames=['filename','ocr_result','correct_value','confidence']); w.writeheader(); w.writerows(rows)
        z.writestr('corrections.csv',text.getvalue().encode('utf-8-sig'))
    return out.getvalue()

def cellok(v): return bool(re.fullmatch(r'[A-Za-z]{1,3}[1-9][0-9]*',v.strip()))

excel=st.file_uploader('1. 入力先Excel',type=['xlsx'])
files=sorted(st.file_uploader('2. 加工前写真を選択',type=['png','jpg','jpeg','webp'],accept_multiple_files=True) or [],key=lambda f:timekey(f.name))
if files: st.info('時間順：'+' → '.join(f.name for f in files))
rows=[]
for i,f in enumerate(files):
    data=f.getvalue()
    try: rgb=np.array(Image.open(io.BytesIO(data)).convert('RGB'))
    except Exception as e: st.error(f'{f.name}を開けません：{e}'); continue
    predicted,conf,logs,previews,ranking=recognize(rgb)
    l,r=st.columns([1,2]); l.image(rgb,caption=f.name,width='stretch')
    corrected=r.text_input('認識結果（未検出・誤認識は修正）', '' if predicted=='未検出' else predicted,key=f'v_{i}_{f.name}',placeholder='例：45.1')
    r.write(f'OCR候補の確度：{conf:.2f}')
    if predicted=='未検出': r.error('未検出：正しい値を入力して修正データとして保存できます。')
    elif conf<.70: r.warning('一致が弱いため確認してください。')
    if r.button('修正データとして保存',key=f's_{i}_{f.name}'):
        if not valid_value(corrected): r.error('20.0～99.9のXX.X形式で入力してください。')
        elif corrected==predicted: r.info('OCR結果と同じため修正データには追加しません。')
        else:
            st.session_state.corrections[f.name]={'ocr':predicted,'correct':corrected,'confidence':round(conf,4),'bytes':data}; r.success('保存しました。')
    with r.expander('詳細を見る'):
        if previews: r.image([cv2.cvtColor(x[1],cv2.COLOR_BGR2RGB) for x in previews],caption=[x[0] for x in previews],width=150)
        shown=0
        for tag,items in logs:
            if items: r.write(tag+'：'+ '、'.join(f'{t}({s:.2f})' for t,s in items)); shown+=1
            if shown>=10: break
        if ranking:r.write('数値候補：'+', '.join(f'{v:.1f}' for _,v,_ in ranking))
    rows.append({'filename':f.name,'value':corrected,'confidence':round(conf,3)})

if st.session_state.corrections:
    st.subheader('保存した修正データ')
    st.dataframe(pd.DataFrame([{'filename':k,'ocr_result':v['ocr'],'correct_value':v['correct'],'confidence':v['confidence']} for k,v in st.session_state.corrections.items()]),width='stretch')
    st.download_button('修正データ一式をダウンロード',correction_zip(),'ocr_corrections.zip','application/zip')
if rows:
    st.dataframe(pd.DataFrame(rows),width='stretch')
    if excel:
        wb=load_workbook(io.BytesIO(excel.getvalue())); sn=st.selectbox('入力シート',wb.sheetnames); start=st.text_input('開始セル','C4')
        if cellok(start):
            letters,r0=coordinate_from_string(start.upper()); col=column_index_from_string(letters); ws=wb[sn]; bad=[]
            for j,row in enumerate(rows):
                try:
                    v=float(row['value']); assert 20<=v<=99.9
                    cell=ws.cell(r0+j,col,v); cell.number_format='0.0'
                except: bad.append(row['filename'])
            if bad:st.error('未検出または無効な値：'+'、'.join(bad))
            else:
                out=io.BytesIO();wb.save(out);end=f'{get_column_letter(col)}{r0+len(rows)-1}';st.success(f'{start.upper()}:{end}へ入力しました');st.download_button('入力済みExcelをダウンロード',out.getvalue(),f'{Path(excel.name).stem}_入力済み.xlsx')
