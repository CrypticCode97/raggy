import { useEffect, useState } from 'react'
import { Check, ChevronLeft, ChevronRight, Cpu, Download, Gauge, Languages, LoaderCircle, Rocket, ShieldCheck, X, Zap } from 'lucide-react'

type Profile = { id:'quality'|'fallback'|'exact'; name:string; description:string; download_gb:number }
type Language = { code:string; name:string; download_mb:number }
type SetupInfo = {
  capabilities:{configured:boolean;device:string;gpu_name:string|null;semantic_runtime:boolean;semantic_runtime_error?:string;profiles:Profile[];ocr_languages:Language[]}
  job:{state:string;stage:string;detail:string;completed:number;total:number;error?:string;restart_required:boolean;configured:boolean}
}

interface SetupWizardProps { open:boolean; onClose:()=>void; onComplete:()=>void }

async function getSetup():Promise<SetupInfo>{const response=await fetch('/api/setup');if(!response.ok)throw new Error('Could not inspect this computer');return response.json()}

export function SetupWizard({open,onClose,onComplete}:SetupWizardProps){
  const [info,setInfo]=useState<SetupInfo|null>(null)
  const [step,setStep]=useState(0)
  const [profile,setProfile]=useState<Profile['id']>('quality')
  const [languages,setLanguages]=useState<string[]>(['eng','deu'])
  const [installRuntime,setInstallRuntime]=useState(true)
  const [error,setError]=useState('')

  useEffect(()=>{if(open)getSetup().then(data=>{setInfo(data);if(!data.capabilities.gpu_name)setProfile('fallback')}).catch(e=>setError(e.message))},[open])
  useEffect(()=>{if(!open||info?.job.state!=='running')return;const timer=window.setInterval(()=>getSetup().then(data=>{setInfo(data);if(data.job.state==='complete')onComplete()}).catch(()=>{}),1000);return()=>window.clearInterval(timer)},[open,info?.job.state,onComplete])
  if(!open)return null
  const caps=info?.capabilities
  const job=info?.job
  const selected=caps?.profiles.find(item=>item.id===profile)
  const ocrDownloadMb=caps?.ocr_languages.filter(item=>languages.includes(item.code)).reduce((total,item)=>total+item.download_mb,0)??0
  const totalDownloadMb=(selected?.download_gb??0)*1000+ocrDownloadMb
  const totalDownload=totalDownloadMb>=1000?`${(totalDownloadMb/1000).toFixed(2)} GB`:`${Math.ceil(totalDownloadMb)} MB`
  const running=job?.state==='running'
  const complete=job?.state==='complete'
  const toggleLanguage=(code:string)=>setLanguages(current=>current.includes(code)?current.filter(item=>item!==code):[...current,code])
  async function begin(){setError('');const response=await fetch('/api/setup',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({profile,languages,install_runtime:installRuntime})});const body=await response.json();if(!response.ok){setError(body.detail||'Setup could not start');return}setInfo(current=>current?{...current,job:body}:current);setStep(3)}
  return <div className="setup-backdrop" role="presentation"><section className="setup-wizard" role="dialog" aria-modal="true" aria-labelledby="setup-title">
    <aside className="setup-rail"><div className="setup-mark"><ShieldCheck size={22}/></div><div><span>RAGGY SETUP</span><h2>Private by design.<br/>Powerful by default.</h2></div><ol>{['Welcome','Search quality','OCR languages','Install'].map((label,index)=><li key={label} className={step===index?'current':step>index?'done':''}><i>{step>index?<Check size={13}/>:index+1}</i>{label}</li>)}</ol><p>Your documents and searches never leave this machine.</p></aside>
    <div className="setup-main">
      {caps?.configured&&!running&&!complete&&<button className="setup-close" onClick={onClose} aria-label="Close setup"><X/></button>}
      {step===0&&<div className="setup-step"><div className="step-icon"><Rocket/></div><span className="kicker">WELCOME TO RAGGY</span><h1 id="setup-title">Let’s prepare your evidence engine.</h1><p>We found the best configuration for this computer. You can review every choice before anything is downloaded.</p><div className="hardware-card"><Cpu/><div><small>DETECTED COMPUTE</small><strong>{caps?.gpu_name||'CPU inference'}</strong><span>{caps?.gpu_name?'GPU acceleration is available':'A lightweight model is recommended'}</span></div><b>{caps?.gpu_name?'CUDA READY':'CPU'}</b></div><div className="privacy-row"><ShieldCheck/><span><strong>Offline after setup</strong>Only model and OCR assets are downloaded.</span></div></div>}
      {step===1&&<div className="setup-step"><span className="kicker">SEARCH QUALITY</span><h1 id="setup-title">Choose your retrieval profile.</h1><p>This controls semantic search and reranking. Exact search is always available.</p><div className="profile-list">{caps?.profiles.map(item=><button key={item.id} className={profile===item.id?'selected':''} onClick={()=>setProfile(item.id)}><i>{item.id==='quality'?<Zap/>:item.id==='fallback'?<Gauge/>:<ShieldCheck/>}</i><span><strong>{item.name}{item.id==='quality'&&caps.gpu_name&&<em>Recommended</em>}</strong><small>{item.description}</small></span><b>{item.download_gb?`${item.download_gb} GB`:'No download'}</b></button>)}</div>{profile!=='exact'&&!caps?.semantic_runtime&&<label className="runtime-choice"><input type="checkbox" checked={installRuntime} onChange={e=>setInstallRuntime(e.target.checked)}/><span><strong>Install or repair the semantic runtime automatically</strong><small>{caps?.semantic_runtime_error||'Includes the accelerator-enabled PyTorch libraries. This is a one-time installation.'}</small></span></label>}</div>}
      {step===2&&<div className="setup-step"><span className="kicker">OCR LANGUAGES</span><h1 id="setup-title">Which languages are in your PDFs?</h1><p>OCR runs only on scanned pages. Select every language you expect to encounter.</p><div className="language-grid">{caps?.ocr_languages.map(language=><label key={language.code} className={languages.includes(language.code)?'selected':''}><input type="checkbox" checked={languages.includes(language.code)} onChange={()=>toggleLanguage(language.code)}/><span>{language.name}<small>{language.code} · {language.download_mb.toFixed(1)} MB</small></span><Check/></label>)}</div><div className="download-summary"><Download/><span><strong>Ready to install · approximately {totalDownload}</strong>{selected?.name} · {languages.length} OCR language{languages.length===1?'':'s'} ({ocrDownloadMb.toFixed(1)} MB)</span></div></div>}
      {step===3&&<div className="setup-step setup-progress">{complete?<><div className="complete-ring"><Check/></div><span className="kicker">SETUP COMPLETE</span><h1 id="setup-title">Your evidence engine is ready.</h1><p>{job?.restart_required?'Restart Raggy once to activate the newly installed accelerator runtime.':'Everything is installed locally and ready to use offline.'}</p><button className="primary wide" onClick={()=>{onComplete();onClose()}}>Start finding evidence <ChevronRight/></button></>:<><div className="progress-ring">{job?.state==='failed'?<X/>:<LoaderCircle className="spin"/>}</div><span className="kicker">{job?.state==='failed'?'ACTION NEEDED':'INSTALLING LOCALLY'}</span><h1 id="setup-title">{job?.stage||'Preparing setup'}</h1><p>{job?.error||job?.detail||'Please keep this window open.'}</p><div className="progress-track"><i style={{width:`${job?.total?job.completed/job.total*100:5}%`}}/></div><small>{job?.completed||0} of {job?.total||0} tasks complete</small>{job?.state==='failed'&&<button className="secondary" onClick={()=>setStep(2)}>Review choices</button>}</>}</div>}
      {error&&<div className="setup-error">{error}</div>}
      {step<3&&<footer><button className="secondary" disabled={step===0} onClick={()=>setStep(value=>value-1)}><ChevronLeft/>Back</button><button className="primary" disabled={!info||(step===2&&!languages.length)} onClick={()=>step===2?begin():setStep(value=>value+1)}>{step===2?'Install locally':'Continue'}<ChevronRight/></button></footer>}
    </div>
  </section></div>
}
