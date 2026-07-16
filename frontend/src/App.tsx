import { FormEvent, useCallback, useEffect, useRef, useState } from 'react'
import { BookOpen, Box, ChevronLeft, ChevronRight, Cpu, FileSearch, FolderPlus, LoaderCircle, Search, Settings, Trash2, X } from 'lucide-react'
import * as pdfjs from 'pdfjs-dist'
import workerSrc from 'pdfjs-dist/build/pdf.worker.min.mjs?url'
import { SetupWizard } from './SetupWizard'

pdfjs.GlobalWorkerOptions.workerSrc = workerSrc

type Mode = 'hybrid' | 'exact' | 'semantic'
type BoxRect = { x:number; y:number; w:number; h:number }
type Result = {page_id:number;document_id:number;title:string;path:string;page_number:number;score:number;snippet:string;highlights:BoxRect[];extraction_method:string;warning?:string;pdf_url:string}
type Source = {id:number;path:string;last_scanned_at?:string}
type Status = {documents:number;pages:number;ocr_languages:string;models:{device:string;profile:string;loaded:boolean};job?:{id:number;state:string;processed_files:number;total_files:number;current_path?:string}}

async function api<T>(url:string, options?:RequestInit):Promise<T>{
  const response=await fetch(url,{headers:{'Content-Type':'application/json',...options?.headers},...options})
  if(!response.ok){const body=await response.json().catch(()=>({detail:response.statusText}));throw new Error(body.detail||'Request failed')}
  return response.json()
}

function PdfViewer({result,onClose}:{result:Result;onClose:()=>void}){
  const canvas=useRef<HTMLCanvasElement>(null); const [page,setPage]=useState(result.page_number); const [pages,setPages]=useState(0); const [loading,setLoading]=useState(true)
  useEffect(()=>setPage(result.page_number),[result])
  useEffect(()=>{let cancelled=false; setLoading(true)
    pdfjs.getDocument(`/api/documents/${result.document_id}/pdf`).promise.then(async doc=>{if(cancelled)return;setPages(doc.numPages);const p=await doc.getPage(page);const viewport=p.getViewport({scale:1.35});const c=canvas.current;if(!c)return;c.width=viewport.width;c.height=viewport.height;await p.render({canvas:c,canvasContext:c.getContext('2d')!,viewport}).promise;setLoading(false)})
    return()=>{cancelled=true}},[result.document_id,page])
  return <div className="viewer-shell">
    <div className="viewer-head"><div><strong>{result.title}</strong><span>Page {page} of {pages||'…'}</span></div><div className="page-controls"><button disabled={page<=1} onClick={()=>setPage(p=>p-1)}><ChevronLeft size={17}/></button><button disabled={page>=pages} onClick={()=>setPage(p=>p+1)}><ChevronRight size={17}/></button><button onClick={onClose}><X size={19}/></button></div></div>
    <div className="canvas-scroll">{loading&&<LoaderCircle className="spin viewer-loader"/>}<div className="canvas-wrap"><canvas ref={canvas}/>{page===result.page_number&&result.highlights.map((b,i)=><i key={i} className="highlight" style={{left:`${b.x*100}%`,top:`${b.y*100}%`,width:`${b.w*100}%`,height:`${b.h*100}%`}}/>)}</div></div>
  </div>
}

export function App(){
  const [query,setQuery]=useState(''); const [mode,setMode]=useState<Mode>('hybrid'); const [results,setResults]=useState<Result[]>([]); const [total,setTotal]=useState<number|null>(null); const [took,setTook]=useState(0); const [busy,setBusy]=useState(false); const [error,setError]=useState(''); const [warning,setWarning]=useState(''); const [selected,setSelected]=useState<Result|null>(null); const [sources,setSources]=useState<Source[]>([]); const [status,setStatus]=useState<Status|null>(null); const [sourcePath,setSourcePath]=useState(''); const [libraryOpen,setLibraryOpen]=useState(false); const [setupOpen,setSetupOpen]=useState(false)
  const refresh=useCallback(()=>Promise.all([api<Source[]>('/api/sources').then(setSources),api<Status>('/api/status').then(setStatus)]).then(()=>undefined).catch(e=>setError(e.message)),[])
  useEffect(()=>{refresh();fetch('/api/setup').then(response=>response.json()).then(data=>{if(!data.capabilities.configured)setSetupOpen(true)});const timer=setInterval(refresh,2000);return()=>clearInterval(timer)},[refresh])
  async function search(e?:FormEvent){e?.preventDefault();if(!query.trim())return;setBusy(true);setError('');try{const data=await api<{results:Result[];total:number|null;took_ms:number;warning?:string}>('/api/search',{method:'POST',body:JSON.stringify({query,mode,page_size:10})});setResults(data.results);setTotal(data.total);setTook(data.took_ms);setWarning(data.warning||'')}catch(e){setError((e as Error).message)}finally{setBusy(false)}}
  async function addSource(e:FormEvent){e.preventDefault();try{await api('/api/sources',{method:'POST',body:JSON.stringify({path:sourcePath})});setSourcePath('');refresh()}catch(e){setError((e as Error).message)}}
  async function index(){try{await api('/api/index-jobs',{method:'POST'});refresh()}catch(e){setError((e as Error).message)}}
  return <div className="app">
    <header><a className="brand"><span><FileSearch size={21}/></span>raggy</a><nav><button className="nav-active">Evidence</button><button onClick={()=>setLibraryOpen(true)}>Library</button></nav><button className="setup-button" onClick={()=>setSetupOpen(true)} title="Setup"><Settings size={16}/>Setup</button><button className="library-button" onClick={()=>setLibraryOpen(true)}><BookOpen size={16}/>{status?.documents||0} documents</button></header>
    <main>
      <section className="hero"><div className="eyebrow"><span/>LOCAL EVIDENCE ENGINE</div><h1>Find the source.<br/><em>Trust the evidence.</em></h1><p>Search technical documents without surrendering provenance. Every result leads back to the original page.</p>
        <form className="searchbox" onSubmit={search}><Search size={21}/><input autoFocus value={query} onChange={e=>setQuery(e.target.value)} placeholder="Search standards, specifications, materials…"/><button disabled={busy}>{busy?<LoaderCircle className="spin" size={19}/>:<>Search <kbd>↵</kbd></>}</button></form>
        <div className="modes">{(['hybrid','exact','semantic'] as Mode[]).map(item=><button key={item} onClick={()=>setMode(item)} className={mode===item?'active':''}><span>{item==='hybrid'?'Recommended':item==='exact'?'Every literal match':'Meaning-based'}</span>{item}</button>)}</div>
      </section>
      {(error||warning)&&<div className={error?'alert error':'alert'}>{error||warning}<button onClick={()=>{setError('');setWarning('')}}><X size={16}/></button></div>}
      <section className="workspace">
        <div className="result-pane"><div className="result-head"><div><h2>{results.length?'Ranked evidence':'Your evidence library'}</h2><p>{results.length?(total!==null?`${total} exact pages · ${took} ms`:`Top ${results.length} pages · ${took} ms`):'Add a folder, build the index, then search naturally or literally.'}</p></div>{status?.job?.state==='running'&&<div className="index-pill"><LoaderCircle className="spin" size={14}/>{status.job.processed_files}/{status.job.total_files}</div>}</div>
          {!results.length?<div className="empty"><div><Box size={28}/></div><h3>Original pages, not generated answers</h3><p>Results preserve the document, page number, extraction method, and the exact matching passage.</p><button onClick={()=>setLibraryOpen(true)}><FolderPlus size={17}/> Add document folder</button></div>:
          <div className="results">{results.map((r,i)=><article key={r.page_id} onClick={()=>setSelected(r)}><div className="rank">{String(i+1).padStart(2,'0')}</div><div className="result-body"><div className="docline"><span>{r.extraction_method==='ocr'?'OCR':'PDF'}</span><strong>{r.title}</strong><small>Page {r.page_number}</small></div><p>{r.snippet}</p><div className="path">{r.path}</div>{r.warning&&<div className="warning">{r.warning}</div>}</div><button className="open-page">Open page <ChevronRight size={16}/></button></article>)}</div>}
        </div>
        <aside><div className="stat"><Cpu size={18}/><div><span>Compute</span><strong>{status?.models.device?.toUpperCase()||'—'} · {status?.models.profile||'—'}</strong></div></div><div className="stat"><BookOpen size={18}/><div><span>Indexed corpus</span><strong>{status?.pages.toLocaleString()||0} pages</strong></div></div><div className="principle"><span>GUIDING PRINCIPLE</span><p>“The system does not answer questions—it finds evidence.”</p></div></aside>
      </section>
    </main>
    {selected&&<div className="modal"><PdfViewer result={selected} onClose={()=>setSelected(null)}/></div>}
    {libraryOpen&&<div className="drawer-backdrop" onClick={()=>setLibraryOpen(false)}><div className="drawer" onClick={e=>e.stopPropagation()}><div className="drawer-head"><div><h2>Evidence library</h2><p>One local index, multiple recursive folders.</p></div><button onClick={()=>setLibraryOpen(false)}><X/></button></div><form onSubmit={addSource}><input value={sourcePath} onChange={e=>setSourcePath(e.target.value)} placeholder="/path/to/pdf/folder"/><button><FolderPlus size={17}/>Add folder</button></form><div className="source-list">{sources.map(s=><div key={s.id}><BookOpen size={17}/><span>{s.path}<small>{s.last_scanned_at?'Last scanned '+s.last_scanned_at:'Not indexed yet'}</small></span><button onClick={async()=>{await fetch(`/api/sources/${s.id}`,{method:'DELETE'});refresh()}}><Trash2 size={16}/></button></div>)}</div><button className="index-button" disabled={!sources.length||status?.job?.state==='running'} onClick={index}>{status?.job?.state==='running'?<><LoaderCircle className="spin"/>Indexing {status.job.processed_files} / {status.job.total_files}</>:<>Build incremental index <ChevronRight/></>}</button><p className="offline-note">OCR languages: {status?.ocr_languages}. Models and OCR assets are used locally.</p></div></div>}
    <SetupWizard open={setupOpen} onClose={()=>setSetupOpen(false)} onComplete={refresh}/>
  </div>
}
