"""Santander Persona Benchmark (V3 PCA/KMeans, V4 FAMD/MFA + micro-Ward).
Source-compatible, frozen batch transforms; full May snapshots are never loaded into RAM.
Missing sidecars fail closed, rather than silently inventing CLV/affinity/lifecycle values.
"""
from __future__ import annotations
import os, re, json, hashlib, warnings, math
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter
import numpy as np
import pandas as pd
import joblib
import duckdb
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score, f1_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
try:
 from scipy.cluster.hierarchy import cut_tree
 from scipy.spatial.distance import cdist
except ImportError as exc:
 if 'scipy._cyutility' not in str(exc):
  raise
 raise ImportError(
  'SciPy compiled extensions are inconsistent in this runtime. Restart the Colab '
  'runtime/kernel after bootstrap, then run from the top. Do not reload or delete '
  'SciPy modules in the current process. If this persists in a fresh runtime, '
  'repair the SciPy installation there before running the benchmark.'
 ) from exc

SEED=42
SNAPSHOT='2016-05-28'
PRODUCTS=['ind_ahor_fin_ult1','ind_aval_fin_ult1','ind_cco_fin_ult1','ind_cder_fin_ult1','ind_cno_fin_ult1','ind_ctju_fin_ult1','ind_ctma_fin_ult1','ind_ctop_fin_ult1','ind_ctpp_fin_ult1','ind_deco_fin_ult1','ind_deme_fin_ult1','ind_dela_fin_ult1','ind_ecue_fin_ult1','ind_fond_fin_ult1','ind_hip_fin_ult1','ind_plan_fin_ult1','ind_pres_fin_ult1','ind_reca_fin_ult1','ind_tjcr_fin_ult1','ind_valo_fin_ult1','ind_viv_fin_ult1','ind_nomina_ult1','ind_nom_pens_ult1','ind_recibo_ult1']
LAG=['prev_'+p for p in PRODUCTS]
BASE_REQUIRED={'ncodpers','fecha_dato','rfm_frequency','rfm_recency_months','rfm_never_acquired_before','rfm_has_previous_record','rfm_monetary','rfm_monetary_rarity_income'}|set(LAG)
RAW_COLUMNS=sorted(set(LAG)|{'ncodpers','fecha_dato','age','income_log','account_age_months','is_new_customer','is_active_customer','employee_status','is_male','is_deceased','country','province','entry_channel','customer_segment','profile_missing_structural','customer_relationship_status','relationship_indrel','relationship_indrel_1mes','relationship_tiprel_1mes','cumulative_drops','record_gap_months','rfm_has_previous_record','rfm_observed_history_records','customer_history_months','rfm_recency_months','rfm_frequency','rfm_never_acquired_before','rfm_monetary','rfm_monetary_rarity_income','rfm_segment_fixed','renta_imputation_method','renta_raw','time_idx','lifecycle_tier','relationship_exit_signal','inactive_streak_months','portfolio_closed_now','recent_drop_now'})
# Frozen exact 49-feature selection extracted from V3-R execution output (20k May run).
PRUNED_49=['age','income_log','account_age_months','is_new_customer','is_active_customer','employee_status','is_male','is_deceased','country','province','entry_channel','customer_segment','customer_relationship_status','cumulative_drops','record_gap_months','rfm_observed_history_records','customer_history_months','rfm_has_previous_record','prev_ind_ahor_fin_ult1','prev_ind_cco_fin_ult1','prev_ind_cder_fin_ult1','prev_ind_cno_fin_ult1','prev_ind_ctju_fin_ult1','prev_ind_ctma_fin_ult1','prev_ind_ctop_fin_ult1','prev_ind_ctpp_fin_ult1','prev_ind_deco_fin_ult1','prev_ind_deme_fin_ult1','prev_ind_dela_fin_ult1','prev_ind_ecue_fin_ult1','prev_ind_fond_fin_ult1','prev_ind_hip_fin_ult1','prev_ind_plan_fin_ult1','prev_ind_pres_fin_ult1','prev_ind_reca_fin_ult1','prev_ind_tjcr_fin_ult1','prev_ind_valo_fin_ult1','prev_ind_viv_fin_ult1','prev_ind_nomina_ult1','prev_ind_nom_pens_ult1','prev_ind_recibo_ult1','record_gap_months_is_missing','rfm_recency_months','rfm_frequency','rfm_monetary_rarity_income','lifecycle_tier','relationship_exit_signal','clv_predicted_acquisitions_3m','clv_predicted_value_3m']
assert len(PRUNED_49)==49
DEMO_NUM=['age','income_log','account_age_months']
DEMO_CAT=['is_new_customer','is_active_customer','employee_status','is_male','is_deceased','country','province','entry_channel','customer_segment','customer_relationship_status','profile_missing_structural']
HIST_NUM=['cumulative_drops','record_gap_months','rfm_observed_history_records','customer_history_months']
HIST_CAT=['rfm_has_previous_record']+LAG
RFM_NUM=['rfm_recency_months','rfm_frequency','rfm_monetary_portfolio','rfm_monetary_rarity_income']
RFM_CAT=['rfm_never_acquired_before','rfm_segment_fixed']
LIFE_CORE=['lifecycle_tier','relationship_exit_signal']
LIFE_OPT=['inactive_streak_months','portfolio_closed_now','recent_drop_now']
CLV_NUM=['clv_predicted_acquisitions_3m','clv_predicted_value_3m']
AFF_NUM=['affinity_to_'+p for p in PRODUCTS]
MISSING_TOKENS={'MISSING','__MISSING__','<MISSING>','NULL','NAN','UNKNOWN','NONE',''}

def qid(s):return '"'+str(s).replace('"','""')+'"'
def qlit(s):return "'"+str(s).replace("'","''")+"'"
def parquet_glob(p):
 p=Path(p)
 return str(p/'**'/'*.parquet') if p.is_dir() else str(p)
def out_json(obj,p):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 p.write_text(json.dumps(obj,indent=2,default=str,ensure_ascii=False),encoding='utf-8')
def read_json(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def setup_db(out):
 con=duckdb.connect(':memory:');con.execute("SET threads=4")
 con.execute(f"SET memory_limit={qlit(os.getenv('SANTANDER_DUCKDB_MEMORY_LIMIT','2GB'))}")
 tmp=Path(out)/'duckdb_tmp';tmp.mkdir(parents=True,exist_ok=True)
 con.execute(f'SET temp_directory={qlit(tmp)}');return con
def schema(con,p):return con.execute(f'DESCRIBE SELECT * FROM read_parquet({qlit(parquet_glob(p))})').df()['column_name'].tolist()
def version_of(p):
 vs=re.findall(r'(?:^|[/_\\-])(?:v|version)[_-]?(\d+)(?=[/_.\\-]|$)',str(p).lower())
 return max(map(int,vs)) if vs else -1

def canonical_resolve(root,con):
 explicit=os.getenv('SANTANDER_CANONICAL_CHECKPOINT','').strip()
 if explicit:
  choices=[Path(explicit)]
 else:
  base=Path(root)/'processed'
  if not base.exists():raise FileNotFoundError(f'Processed data missing: {base}')
  choices=[p for p in base.rglob('*.parquet') if not any(t in str(p.relative_to(base)).lower() for t in ('persona','affinity','clv/','lifecycle','eda/','rfm_eda','submission','predictions'))]
 rows=[]
 for p in choices:
  if not p.exists():continue
  try:
   col=set(schema(con,p));missing=BASE_REQUIRED-col
   rows.append(dict(path=str(p),version=version_of(p),valid=not missing,missing=','.join(sorted(missing)[:8])))
  except Exception as e:rows.append(dict(path=str(p),version=version_of(p),valid=False,missing=str(e)[:140]))
 tab=pd.DataFrame(rows)
 if tab.empty or not tab.valid.any():raise FileNotFoundError('No schema-valid canonical checkpoint; set SANTANDER_CANONICAL_CHECKPOINT. Candidate audit: '+str(rows[:5]))
 ok=tab[tab.valid].sort_values(['version','path'],ascending=[False,True]);chosen=Path(ok.iloc[0].path)
 if explicit and not chosen.exists():raise FileNotFoundError(f'Explicit checkpoint not found: {chosen}')
 return chosen,tab

def make_canonical_stage(root,out,con,force=False,canonical_path=None):
 """DuckDB out-of-core May stage and strict canonical source metadata; never recompute RFM."""
 out=Path(out);stage=out/'shared'/'may_canonical_thin.parquet';meta=stage.with_suffix('.meta.json')
 if canonical_path is None:
  canonical,audit=canonical_resolve(root,con)
 else:
  canonical=Path(canonical_path)
  if not canonical.exists():
   raise FileNotFoundError(f'Canonical from template-compatible resolver is missing: {canonical}')
  audit=pd.DataFrame([dict(path=str(canonical),version=version_of(canonical),valid=True)])
 cs=set(schema(con,canonical))
 missing=BASE_REQUIRED-cs
 if missing:raise ValueError(f'Canonical missing required: {sorted(missing)}')
 cols=[c for c in RAW_COLUMNS if c in cs];src=parquet_glob(canonical)
 st=canonical.stat()
 provenance=dict(canonical=str(canonical.resolve()),source_size=st.st_size if canonical.is_file() else None,source_mtime_ns=st.st_mtime_ns,columns=cols,snapshot=SNAPSHOT)
 if not force and stage.exists() and meta.exists() and read_json(meta)==provenance:return stage,canonical,audit
 stage.parent.mkdir(parents=True,exist_ok=True)
 projection=', '.join('m.'+qid(c) for c in cols)
 history_join=''
 if 'customer_history_months' not in cols:
  history_join=f"LEFT JOIN (SELECT ncodpers, MIN(CAST(fecha_dato AS DATE)) AS first_seen FROM read_parquet({qlit(src)}) WHERE CAST(fecha_dato AS DATE)<=DATE '2016-05-28' GROUP BY 1) h USING(ncodpers)"
  projection+=", date_diff('month',h.first_seen,CAST(m.fecha_dato AS DATE)) AS customer_history_months"
 query=f"SELECT {projection} FROM read_parquet({qlit(src)}) m {history_join} WHERE date_trunc('month',CAST(m.fecha_dato AS DATE))=DATE '2016-05-01'"
 temp=stage.with_name('may_canonical_thin.tmp.parquet');temp.unlink(missing_ok=True)
 con.execute(f"COPY ({query}) TO {qlit(temp)} (FORMAT PARQUET,COMPRESSION ZSTD,ROW_GROUP_SIZE 10000)")
 count,unique=con.execute(f'SELECT COUNT(*),COUNT(DISTINCT ncodpers) FROM read_parquet({qlit(temp)})').fetchone()
 if count==0 or count!=unique:
  temp.unlink(missing_ok=True);raise ValueError(f'May row/customer key issue: {count} rows {unique} unique')
 temp.replace(stage);out_json(provenance,meta);return stage,canonical,audit

def resolve_sidecar(root,env,relative):
 explicit=os.getenv(env,'').strip()
 choices=[Path(explicit)] if explicit else [Path(root)/'processed'/relative,Path('/content/drive/MyDrive/projects/santander/data/processed')/relative]
 for p in choices:
  if p.exists():return p
 raise FileNotFoundError(f'{env} missing. Searched {[str(c) for c in choices]}')

def find_order(obj):
 if isinstance(obj,dict):
  for k in ('product_columns','products','product_order','target_columns'):
   x=obj.get(k)
   if isinstance(x,list) and len(x)==24:return x
  for v in obj.values():
   z=find_order(v)
   if z:return z
 return None

class Sidecars:
 def __init__(self,root,con):
  self.con=con;self.root=Path(root)
  self.clv=resolve_sidecar(root,'SANTANDER_CLV_RUN_DIR',Path('CLV')/'clv_portfolio_rfm_u86rd2lo')
  if 'clv_portfolio_rfm_u86rd2lo' not in str(self.clv) and os.getenv('SANTANDER_CLV_RUN_VERIFIED','0')!='1':
   raise ValueError('CLV run path differs from source validated run ID. Set SANTANDER_CLV_RUN_VERIFIED=1 after inspecting provenance.')
  self.affdir=resolve_sidecar(root,'SANTANDER_AFFINITY_DIR',Path('affinity_benchmark')/'v1_customer_month')
  ap=self.affdir/'temporal_affinity.npz'
  if not ap.exists():raise FileNotFoundError(f'Affinity missing {ap}')
  with np.load(ap,allow_pickle=False) as z:
   if not {'affinity','N','D','prior'}.issubset(z.files):raise ValueError('Affinity missing required keys')
   self.A=np.asarray(z['affinity'],dtype=np.float64)
  if self.A.shape!=(24,24) or not np.isfinite(self.A).all():raise ValueError('Invalid affinity 24x24')
  np.fill_diagonal(self.A,0)
  override=os.getenv('SANTANDER_AFFINITY_PRODUCT_ORDER','').strip();self.order=None
  if override:
   if Path(override).is_file():self.order=read_json(override)
   else:self.order=[x.strip() for x in override.split(',')]
  else:
   for fn in ('temporal.json','benchmark_summary.json'):
    p=self.affdir/fn
    if p.exists():self.order=find_order(read_json(p))
    if self.order:break
  if self.order is None:raise ValueError('Affinity product order UNVERIFIED. Supply SANTANDER_AFFINITY_PRODUCT_ORDER; do not guess.')
  if len(self.order)!=24 or set(self.order)!=set(PRODUCTS):raise ValueError('Affinity product order incompatible with canonical 24 products')
  explicit=os.getenv('SANTANDER_LIFECYCLE_PATH','').strip()
  if explicit:self.lifecycle=Path(explicit)
  else:
   options=[]
   for base in [self.root/'processed'/'EDA',self.root/'processed'/'lifecycle']:
    if base.exists():options.extend(p for p in base.rglob('*.parquet') if 'lifecycle' in p.name.lower())
   options=[p for p in options if {'ncodpers','fecha_dato',*LIFE_CORE}.issubset(schema(con,p))]
   if not options:self.lifecycle=None
   else:self.lifecycle=sorted(options,key=lambda p:(version_of(p),p.stat().st_mtime),reverse=True)[0]
  self.pred=self.clv/'reference_predictions_portfolio.parquet';self.ref=self.clv/'reference_asof.parquet'
  self.predcols=set(schema(con,self.pred)) if self.pred.exists() else set()
  self.predacq=next((x for x in ('pred_acq_ml_stable','pred_acquisitions_ml_stable','pred_count_ml_stable') if x in self.predcols),None)
  self.predval=next((x for x in ('pred_monetary_ml_stable',) if x in self.predcols),None)
  self.boosters=None
  if not self.predacq or not self.predval:
   import lightgbm as lgb
   self.boosters=[lgb.Booster(model_file=str(self.clv/name)) for name in ('lgbm_acquisition_stable_cutoff_20160528_h3m.txt','lgbm_monetary_stable_cutoff_20160528_h3m.txt')]
   expected=['never_acquired','acquisition_rate','observed_history_months_log1p']
   for b in self.boosters:
    if b.feature_name()!=expected:raise ValueError('Stable CLV saved model feature order mismatch')
  self.lifecols=set(schema(con,self.lifecycle)) if self.lifecycle else set()
  self.info=dict(clv=str(self.clv),clv_predictions=str(self.pred),affinity=str(ap),
                 affinity_product_order=self.order,affinity_metadata=str(self.affdir/'temporal.json'),
                 lifecycle=str(self.lifecycle) if self.lifecycle else 'canonical',
                 cutoff_assumption='Affinity generated with data through Apr 2016; validate upstream artifact metadata')

 def _subset(self,path,ids,cols,key='ncodpers',month='fecha_dato',observed_only=False):
  if not cols:return pd.DataFrame({'ncodpers':ids.ncodpers.to_numpy()})
  db=self.con;names=set(schema(db,path));alias_key='customer_id' if 'customer_id' in names else key
  alias_month='cutoff_month' if 'cutoff_month' in names else month
  if not set(cols+[alias_key,alias_month]).issubset(names):raise ValueError(f'{path}: missing required columns {set(cols+[alias_key,alias_month])-names}')
  temp=ids[['ncodpers']].copy();db.register('persona_ids',temp)
  sql=f"SELECT s.ncodpers,{','.join('p.'+qid(c) for c in cols)} FROM read_parquet({qlit(parquet_glob(path))}) p JOIN persona_ids s ON TRY_CAST(p.{qid(alias_key)} AS BIGINT)=s.ncodpers WHERE date_trunc('month',CAST(p.{qid(alias_month)} AS DATE))=DATE '2016-05-01'"
  if observed_only and 'record_type' in names:sql+=" AND LOWER(CAST(p.record_type AS VARCHAR)) NOT LIKE '%synthetic%'"
  try:res=db.execute(sql).df()
  finally:db.unregister('persona_ids')
  if res.ncodpers.duplicated().any():raise ValueError(f'Duplicate customer-month in sidecar {path}')
  return res

 def enrich(self,base):
  """Same as-of semantic joins for sample and streamed production batches."""
  f=base.copy();f['ncodpers']=pd.to_numeric(f.ncodpers,errors='raise').astype('int64')
  if f.ncodpers.duplicated().any():raise ValueError('Duplicate customer IDs')
  if 'customer_relationship_status' not in f:
   rel=['relationship_indrel','relationship_indrel_1mes','relationship_tiprel_1mes']
   if not set(rel).issubset(f):raise ValueError('Canonical relationship status/mapping missing')
   f['customer_relationship_status']=f[rel].fillna('UNKNOWN').astype(str).agg('|'.join,axis=1)
   warnings.warn('Using raw relationship tuple as legacy fallback; not a verified canonical semantic mapping.')
  f['rfm_monetary_portfolio']=f['rfm_monetary']
  clv_cols=['clv_predicted_acquisitions_3m','clv_predicted_value_3m']
  if not set(clv_cols).issubset(f):
   if self.predacq and self.predval:
    cached=self._subset(self.pred,f,[self.predacq,self.predval])
    cached=cached.rename(columns={self.predacq:clv_cols[0],self.predval:clv_cols[1]})
   else:
    ref=self._subset(self.ref,f,['total_events','observed_history_months'])
    if len(ref)!=len(f):raise ValueError('Incomplete CLV reference_asof coverage')
    tot=ref.total_events.to_numpy(dtype=float);m=ref.observed_history_months.to_numpy(dtype=float)
    if not np.isfinite(tot).all() or not np.isfinite(m).all() or (m<0).any():raise ValueError('Invalid CLV reference')
    inp=pd.DataFrame({'never_acquired':(tot==0).astype('int8'),'acquisition_rate':tot/(m+1),'observed_history_months_log1p':np.log1p(m)})
    cached=ref[['ncodpers']].copy()
    for c,b in zip(clv_cols,self.boosters):cached[c]=b.predict(inp)
   if len(cached)!=len(f):raise ValueError('CLV incomplete coverage for requested customers')
   f=f.merge(cached,on='ncodpers',validate='one_to_one',how='left')
  if f[clv_cols].isna().any().any() or not np.isfinite(f[clv_cols].to_numpy(dtype=float)).all():raise ValueError('CLV missing/nonfinite; repair source')
  f['clv_prediction_available']=1
  history=pd.to_numeric(f.rfm_has_previous_record,errors='coerce').fillna(0).eq(1).to_numpy()
  B=f[['prev_'+p for p in self.order]].apply(pd.to_numeric,errors='coerce').to_numpy(dtype=float)
  if np.any(history & (~np.isfinite(B).all(axis=1) | ~np.isin(np.nan_to_num(B,nan=-1),[0,1]).all(axis=1))):raise ValueError('Invalid known history product states')
  B=np.nan_to_num(B,nan=0);B[~history,:]=0
  scores=(B@self.A)/np.maximum(B.sum(axis=1)[:,None],1)
  scores[~history,:]=np.nan
  for i,p in enumerate(self.order):f['affinity_to_'+p]=scores[:,i].astype('float32')
  f['affinity_available']=history.astype('int8')
  required=[c for c in LIFE_CORE if c not in f]
  optional=[c for c in LIFE_OPT if c not in f and c in self.lifecols]
  if required or optional:
   if not self.lifecycle:raise FileNotFoundError(f'Lifecycle sidecar missing; required {required}. Set SANTANDER_LIFECYCLE_PATH')
   l=self._subset(self.lifecycle,f,required+optional,observed_only=True)
   f=f.merge(l,on='ncodpers',validate='one_to_one',how='left')
  if not set(LIFE_CORE).issubset(f):raise ValueError('Lifecycle required features unavailable')
  return f

def sample_ids(con,stage,n_fit=20000,n_eval=5000,n_anchor=10000):
 """Stable, disjoint samples; no five-way partition. Hash order is independent of row order."""
 n=int(con.execute(f'SELECT COUNT(*) FROM read_parquet({qlit(stage)})').fetchone()[0]);sizes=[n_fit,n_eval,n_anchor]
 if sum(sizes)>n:
  ratio=n/sum(sizes);sizes=[max(10,int(x*ratio)) for x in sizes]
  while sum(sizes)>n:sizes[np.argmax(sizes)]-=1
 ids=con.execute(f'SELECT ncodpers FROM read_parquet({qlit(stage)}) ORDER BY hash(ncodpers, 42) LIMIT {sum(sizes)}').df()
 if len(ids)!=sum(sizes):raise ValueError('Sampling count mismatch')
 parts={};start=0
 for name,count in zip(['train','eval','anchor'],sizes):
  parts[name]=ids.ncodpers.iloc[start:start+count].tolist();start+=count
 return parts

def fetch_ids(con,stage,ids):
 con.register('persona_selected_ids',pd.DataFrame({'ncodpers':pd.Series(ids,dtype='int64')}))
 try:f=con.execute(f'SELECT p.* FROM read_parquet({qlit(stage)}) p JOIN persona_selected_ids s USING(ncodpers) ORDER BY hash(p.ncodpers,42)').df()
 finally:con.unregister('persona_selected_ids')
 if len(f)!=len(ids):raise ValueError(f'Sample missing keys: {len(f)} vs {len(ids)}')
 return f

def registry(frame):
 blocks={'Demographics_num':DEMO_NUM,'Demographics_cat':DEMO_CAT,
  'History_num':HIST_NUM,'History_cat':HIST_CAT,
  'RFM_num':RFM_NUM,'RFM_cat':RFM_CAT,
  'Lifecycle_num':[x for x in ['inactive_streak_months'] if x in frame],
  'Lifecycle_cat':LIFE_CORE+[x for x in ['portfolio_closed_now','recent_drop_now'] if x in frame],
  'CLV_num':CLV_NUM,'CLV_cat':['clv_prediction_available'],
  'Affinity_num':AFF_NUM,'Affinity_cat':['affinity_available']}
 allnames=[c for vals in blocks.values() for c in vals]
 if len(allnames)!=len(set(allnames)):raise ValueError('Duplicate registry feature')
 missing=set(allnames)-set(frame.columns)
 if missing:raise ValueError(f'Unavailable features: {sorted(missing)}')
 rows=[dict(feature=c,group=g.split('_')[0],type=('num' if g.endswith('_num') else 'cat')) for g,names in blocks.items() for c in names]
 rows.append(dict(feature='record_gap_months_is_missing',group='History',type='cat',generated=True))
 return pd.DataFrame(rows).fillna({'generated':False})

def missing_mask(series):
 s=series.astype('string').str.strip().str.upper()
 return series.isna()|s.isin(MISSING_TOKENS)

def add_missing_flag(frame):
 f=frame.copy()
 f['record_gap_months_is_missing']=np.where(missing_mask(f['record_gap_months']),'YES','NO')
 return f

class FrozenPrep:
 """One consistent fit-on-train-only set of categorical imputation/rare levels and numeric rules."""
 def __init__(self,feature_list,reg,rare_min=5,max_levels=40,group_balance=True):
  self.features=list(feature_list);self.reg=reg.set_index('feature')
  self.nums=[x for x in self.features if self.reg.loc[x,'type']=='num']
  self.cats=[x for x in self.features if self.reg.loc[x,'type']=='cat']
  self.rare_min=rare_min;self.max_levels=max_levels;self.group_balance=group_balance
 def fit(self,data):
  f=add_missing_flag(data);self.medians={};self.log_rules={};self.cat_keep={}
  for c in self.nums:
   x=pd.to_numeric(f[c].mask(missing_mask(f[c])),errors='coerce').replace([np.inf,-np.inf],np.nan)
   if x.notna().sum()==0:raise ValueError(f'All missing numeric: {c}')
   self.medians[c]=float(x.median());x=x.fillna(self.medians[c])
   skew=float(x.skew())
   # Keep canonical log-income as-is. Only skewed, non-binary count/continuous fields.
   self.log_rules[c]=('log1p' if c!='income_log' and abs(skew)>=1 and x.min()>=0 else
                     'signed_log1p' if c!='income_log' and abs(skew)>=1 and x.min()<0 else 'identity')
  for c in self.cats:
   x=f[c].astype('string').str.strip().mask(missing_mask(f[c]),'UNKNOWN').fillna('UNKNOWN').astype(str)
   freq=x.value_counts()
   keep=[k for k,n in freq.items() if n>=self.rare_min][:self.max_levels-1]
   if 'UNKNOWN' in freq.index and 'UNKNOWN' not in keep:keep=keep[:self.max_levels-2]+['UNKNOWN']
   self.cat_keep[c]=set(keep)
  X=self.transform_frame(data)
  self.constant=[c for c in self.features if X[c].nunique(dropna=False)<=1]
  self.active=[c for c in self.features if c not in self.constant]
  self.nums=[c for c in self.nums if c in self.active]
  self.cats=[c for c in self.cats if c in self.active]
  X=X[self.active]
  self.scaler=StandardScaler().fit(X[self.nums].to_numpy(dtype=np.float64)) if self.nums else None
  cats=[sorted(set(X[c].astype(str))|{'UNKNOWN','OTHER'}) for c in self.cats]
  self.encoder=OneHotEncoder(categories=cats,handle_unknown='ignore',dtype=np.float32,sparse_output=False) if self.cats else None
  if self.encoder:self.encoder.fit(X[self.cats].astype(str))
  self.column_group=[self.reg.loc[c,'group'] for c in self.nums]
  for c,categories in zip(self.cats, self.encoder.categories_ if self.encoder else []):self.column_group.extend([self.reg.loc[c,'group']]*len(categories))
  mat=self._unweighted(data)
  var=np.var(mat,axis=0,dtype=np.float64)
  self.group_var={g:float(var[np.array(self.column_group)==g].sum()) for g in set(self.column_group)}
  nonzero=[v for v in self.group_var.values() if v>1e-12];goal=float(np.mean(nonzero)) if nonzero else 1.0
  self.weights={g:float(np.clip(np.sqrt(goal/max(v,1e-12)),.25,4)) if self.group_balance else 1.0 for g,v in self.group_var.items()}
  self.weight_vec=np.asarray([self.weights[g] for g in self.column_group],dtype=np.float32)
  return self
 def transform_frame(self,data):
  f=add_missing_flag(data);out=pd.DataFrame(index=f.index)
  for c in self.nums:
   x=pd.to_numeric(f[c].mask(missing_mask(f[c])),errors='coerce').replace([np.inf,-np.inf],np.nan).fillna(self.medians[c]).astype(float)
   rule=self.log_rules[c]
   if rule=='log1p':x=np.log1p(np.maximum(x,0))
   elif rule=='signed_log1p':x=np.sign(x)*np.log1p(np.abs(x))
   out[c]=x.astype('float32')
  for c in self.cats:
   x=f[c].astype('string').str.strip().mask(missing_mask(f[c]),'UNKNOWN').fillna('UNKNOWN').astype(str)
   out[c]=x.where(x.isin(self.cat_keep[c]),'OTHER').astype(str)
  # Before .fit completes, cols includes constant, later only active.
  return out
 def _unweighted(self,data):
  f=self.transform_frame(data)
  a=self.scaler.transform(f[self.nums].to_numpy(dtype=float)).astype('float32') if self.nums else np.empty((len(f),0),dtype='float32')
  b=self.encoder.transform(f[self.cats].astype(str)).astype('float32') if self.cats else np.empty((len(f),0),dtype='float32')
  return np.ascontiguousarray(np.hstack([a,b]),dtype=np.float32)
 def encode(self,data):return self._unweighted(data)*self.weight_vec[None,:]
 def famd_raw(self,data):
  f=self.transform_frame(data)
  # Both FAMD and MFA consume the exact same transformed numerical+categorical view.
  return f[self.active].copy()
 def summary(self):
  return dict(n_original=len(self.features),active=self.active,constant=self.constant,numerical=self.nums,categorical=self.cats,skew_rules=self.log_rules,group_variance=self.group_var,group_weights=self.weights)

def prepare_registry(train):
 reg=registry(train);full=reg.feature.tolist();pruned=PRUNED_49.copy()
 if not set(pruned).issubset(set(full)):raise AssertionError(f'Pruned registry missing {set(pruned)-set(full)}')
 # V3-R has three constant columns on its 20k sample. Confirm with the new fit,
 # rather than hardcoding their removal (sample-dependent).
 return reg,{'Full':full,'Pruned49':pruned}

class FrozenFAMD:
 """Matches uploaded V4 FAMD-style normalized dummy PCA, not prince.FAMD API."""
 def __init__(self,n_components=80,seed=SEED):self.n_components=n_components;self.seed=seed
 def fit(self,data,nums,cats):
  self.nums=list(nums);self.cats=list(cats);X=data[self.nums].to_numpy(dtype=float)
  self.means=X.mean(axis=0);self.stds=X.std(axis=0);self.stds[self.stds<1e-12]=1
  dummy=pd.get_dummies(data[self.cats].astype(str),dtype=float,prefix_sep='__') if self.cats else pd.DataFrame(index=data.index)
  self.dummy_columns=list(dummy.columns);self.prop=np.maximum(dummy.mean(axis=0).to_numpy(dtype=float),1e-10)
  z=self._raw(data);maximum=min(z.shape[0]-1,z.shape[1],self.n_components)
  if maximum<1:raise ValueError('No FAMD components available')
  # randomised for CPU, full when requesting almost all dimensions
  solver='full' if maximum>=min(z.shape)-1 else 'randomized'
  self.pca=PCA(n_components=maximum,svd_solver=solver,random_state=self.seed).fit(z)
  self.ratio=self.pca.explained_variance_ratio_
  return self
 def _raw(self,data):
  X=data[self.nums].to_numpy(dtype=float);x=(X-self.means)/self.stds
  cat=pd.get_dummies(data[self.cats].astype(str),dtype=float,prefix_sep='__') if self.cats else pd.DataFrame(index=data.index)
  cat=cat.reindex(columns=self.dummy_columns,fill_value=0)
  return np.concatenate([x,(cat.to_numpy(dtype=float)-self.prop)/np.sqrt(self.prop)],axis=1)
 def project(self,data):return self.pca.transform(self._raw(data)).astype(np.float32)

class FactorModel:
 def __init__(self,method,pre,max_components=128,seed=SEED):self.method=method;self.pre=pre;self.max_components=max_components;self.seed=seed
 def fit(self,train):
  frame=self.pre.famd_raw(train)
  if self.method=='PCA':
   X=self.pre.encode(train);d=min(self.max_components,X.shape[0]-1,X.shape[1]);self.proj=PCA(n_components=d,svd_solver='full' if d>=min(X.shape)-1 else 'randomized',random_state=self.seed).fit(X)
   self.ratio=self.proj.explained_variance_ratio_;self.fit_z=self.proj.transform(X).astype(np.float32)
  elif self.method=='FAMD':
   self.proj=FrozenFAMD(self.max_components,self.seed).fit(frame,self.pre.nums,self.pre.cats)
   self.ratio=self.proj.ratio;self.fit_z=self.proj.project(frame)
  elif self.method=='MFA':
   import prince
   groups={g:[c for c in self.pre.active if self.pre.reg.loc[c,'group']==g] for g in sorted(set(self.pre.reg.loc[self.pre.active,'group']))}
   # Match original notebook technical Num/Cat blocks, not a single group per business.
   grp={}
   for group in sorted(set(self.pre.reg.loc[self.pre.active,'group'])):
    for suffix,cols in (('_num',self.pre.nums),('_cat',self.pre.cats)):
     names=[c for c in cols if self.pre.reg.loc[c,'group']==group]
     if names:grp[group+suffix]=names
   self.groups=grp;ordered=[c for names in grp.values() for c in names]
   self.mfa_cols=ordered
   mf=frame[ordered].copy();mf.columns=pd.MultiIndex.from_tuples([(g,c) for g,cs in grp.items() for c in cs],names=['group','feature'])
   d=min(self.max_components,len(mf)-1,len(ordered)+sum(frame[c].nunique() for c in self.pre.cats))
   # Prince 0.14--0.16 does not expose PCA's rescale_* constructor options.
   # Let MFA apply its own group normalization instead of forwarding PCA kwargs.
   self.proj=prince.MFA(n_components=d,n_iter=3,copy=False,check_input=False,engine='sklearn',random_state=self.seed)
   self.proj.fit(mf,groups=list(grp));self.fit_z=self.proj.row_coordinates(mf).to_numpy(dtype=np.float32)
   try:self.ratio=np.asarray(self.proj.percentage_of_variance_,dtype=float)/100
   except Exception:self.ratio=np.full(self.fit_z.shape[1],np.nan)
  else:raise ValueError(self.method)
  return self
 def project(self,data):
  if self.method=='PCA':return self.proj.transform(self.pre.encode(data)).astype(np.float32)
  frame=self.pre.famd_raw(data)
  if self.method=='FAMD':return self.proj.project(frame)
  mf=frame[self.mfa_cols].copy();mf.columns=pd.MultiIndex.from_tuples([(g,c) for g,cs in self.groups.items() for c in cs],names=['group','feature'])
  return self.proj.row_coordinates(mf).to_numpy(dtype=np.float32)
 def dims(self,choices=(5,8,10,0.95)):
  out={}
  for d in choices:
   if d==.95:
    ratio=np.asarray(self.ratio,dtype=float)
    if np.isfinite(ratio).all() and float(ratio.sum())>=.95-1e-5:
     out['95%']=int(np.searchsorted(np.cumsum(ratio),.95)+1)
    else:out['95%']=None
   elif int(d)<=self.fit_z.shape[1]:out[str(d)]=int(d)
   else:out[str(d)]=None
  return out

def weighted_ward_linkage(centers,weights):
 from scipy.spatial.distance import cdist
 centers=np.asarray(centers,dtype=np.float64);weights=np.asarray(weights,dtype=np.float64)
 m=len(weights);mass=np.zeros(2*m-1);mass[:m]=weights
 leaves=np.zeros(2*m-1,int);leaves[:m]=1
 delta=np.full((2*m-1,2*m-1),np.inf)
 dist=cdist(centers,centers,'sqeuclidean')
 delta[:m,:m]=np.outer(weights,weights)/(weights[:,None]+weights[None,:])*dist
 np.fill_diagonal(delta,np.inf);active=set(range(m));links=[]
 for step in range(m-1):
  ids=np.asarray(sorted(active),int);sub=delta[np.ix_(ids,ids)]
  u,v=np.unravel_index(int(np.argmin(sub)),sub.shape);a,b=sorted([int(ids[u]),int(ids[v])]);value=max(0,float(delta[a,b]));new=m+step
  mass[new]=mass[a]+mass[b];leaves[new]=leaves[a]+leaves[b]
  links.append([a,b,np.sqrt(2*value),leaves[new]])
  for c in active-{a,b}:
   upd=((mass[a]+mass[c])*delta[a,c]+(mass[b]+mass[c])*delta[b,c]-mass[c]*value)/(mass[a]+mass[b]+mass[c])
   delta[new,c]=delta[c,new]=max(0,upd)
  active.remove(a);active.remove(b);active.add(new)
 return np.asarray(links,dtype=float)

class WardMicro:
 def __init__(self,d,k,micro=128,seed=SEED):self.d=d;self.k=k;self.micro=micro;self.seed=seed
 def fit(self,z):
  x=z[:,:self.d];m=min(self.micro,max(2,len(x)//15),len(x)-1)
  self.km=KMeans(n_clusters=m,n_init=5,random_state=self.seed,max_iter=200)
  micro_ids=self.km.fit_predict(x);self.mass=np.bincount(micro_ids,minlength=m)
  if (self.mass==0).any():raise ValueError('Empty HCPC microcluster')
  self.ward=weighted_ward_linkage(self.km.cluster_centers_,self.mass)
  self.mapping=cut_tree(self.ward,n_clusters=[self.k]).reshape(-1).astype('int16')
  if len(np.unique(self.mapping))!=self.k:raise ValueError('HCPC cut_tree invalid')
  return self
 def predict(self,z):return self.mapping[self.km.predict(z[:,:self.d])]

class PersonaModel:
 def __init__(self,feature_set,method,d_label,d,k,pre,factor,cluster):
  self.feature_set=feature_set;self.method=method;self.d_label=d_label;self.d=d;self.k=k
  self.pre=pre;self.factor=factor;self.cluster=cluster
 def project(self,data):return self.factor.project(data)[:,:self.d]
 def predict(self,data):
  z=self.project(data)
  return self.cluster.predict(z)
 def metadata(self):return dict(feature_set=self.feature_set,method=self.method,d_label=self.d_label,d=self.d,k=self.k)

def fit_model(data,registry,features,method,d_label,k,seed=SEED,max_components=128,micro=128,group_balance=True):
 # CSV readers infer integer labels when a grid has no available "95%" row.
 d_label=str(d_label)
 pre=FrozenPrep(features,registry,group_balance=group_balance).fit(data)
 factor=FactorModel(method,pre,max_components,seed).fit(data)
 dims=factor.dims();d=dims.get(d_label)
 if d is None:raise ValueError(f'Unavailable dimension {d_label}: available ratio {np.nansum(factor.ratio):.3f}')
 z=factor.fit_z[:,:d]
 if method=='PCA':cl=KMeans(n_clusters=k,random_state=seed,n_init=10,max_iter=300).fit(z)
 else:cl=WardMicro(d,k,micro,seed).fit(z)
 return PersonaModel('',method,d_label,d,k,pre,factor,cl)

def score_silhouette(z,labels,indices):
 yy=np.asarray(labels)[indices];zz=np.asarray(z)[indices]
 if len(np.unique(yy))<2 or len(np.unique(yy))>=len(yy):return np.nan
 return float(silhouette_score(zz,yy,metric='euclidean'))

def stable_seed_samples(n,rounds=5,fraction=.8,seed=SEED):
 rng=np.random.default_rng(seed)
 return [np.sort(rng.choice(n,size=max(100,int(n*fraction)),replace=False)) for _ in range(rounds)]

def safe_cat_vs_cluster(frame,columns,y):
 from sklearn.metrics import normalized_mutual_info_score
 output=[]
 for col in columns:
  if col in frame:
   x=frame[col].astype('string').fillna('UNKNOWN').astype(str)
   output.append(dict(feature=col,nmi=float(normalized_mutual_info_score(x,y))))
 return pd.DataFrame(output,columns=['feature','nmi']).sort_values('nmi',ascending=False)
