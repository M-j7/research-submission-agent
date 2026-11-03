import { Children, isValidElement, useState } from 'react'
import type { ReactNode } from 'react'
import ReactMarkdown from 'react-markdown'

function safeLink(url: string) {
  try { const u=new URL(url); return ['http:','https:'].includes(u.protocol) && !u.username && !u.password ? u.href : undefined }
  catch {return undefined}
}
function CodeBlock({children}:{children?:ReactNode}) {
  const [label,setLabel]=useState('复制代码')
  const code=Children.toArray(children).map(child=>isValidElement<{children?:ReactNode}>(child)?String(child.props.children??''):String(child)).join('')
  return <div className="code-block"><button type="button" onClick={async()=>{
    try {await navigator.clipboard.writeText(code);setLabel('已复制')}catch{setLabel('复制失败，请手动选择')}
  }}>{label}</button><pre>{children}</pre></div>
}
export function SafeMarkdown({text}:{text:string}) {
  return <div className="markdown"><ReactMarkdown skipHtml urlTransform={url=>safeLink(url)||''}
    components={{a:({href,children})=>href?<a href={href} target="_blank" rel="noopener noreferrer">{children}</a>:<span>{children}</span>,
      img:({alt})=><span>{alt ? `[图片：${alt}]`:'[图片已省略]'}</span>,
      pre:({children})=><CodeBlock>{children}</CodeBlock>}}>{text}</ReactMarkdown></div>
}
