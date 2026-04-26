import React, { useEffect, useMemo, useRef, useState } from 'react';
import { askCreativeChat } from '../services/api';

const LANGUAGE_OPTIONS = [
  { value: 'catalan', label: 'Cat' },
  { value: 'castilian', label: 'ES' },
  { value: 'english', label: 'EN' },
];

const COPY = {
  catalan: {
    greeting: (brand) => `Ei, soc el Creative Coach de ${brand}. Pregunta'm el que vulguis sobre punts febles, millores visuals i decisions de disseny.`,
    hint: 'Pregunta coses concretes del creative',
    starters: [
      'Que li falta visualment?',
      'Com el milloraries en 3 canvis?',
      'On esta el CTA i es prou visible?',
    ],
    thinking: 'pensant...',
    placeholder: 'Ex: on falla aquest creative?',
    send: 'Enviar',
    implement: '⚡ Implementar',
    implementing: 'Implementant...',
    implemented: 'Canvi aplicat! Vols veure-ho?',
  },
  castilian: {
    greeting: (brand) => `Hola, soy el Creative Coach de ${brand}. Preguntame sobre puntos debiles y mejoras visuales.`,
    hint: 'Pregunta sobre el creative',
    starters: [
      'Que le falta visualmente?',
      'Como lo mejorarias en 3 cambios?',
      'Donde esta el CTA?',
    ],
    thinking: 'pensando...',
    placeholder: 'Ej: donde falla este creative?',
    send: 'Enviar',
    implement: '⚡ Implementar',
    implementing: 'Implementando...',
    implemented: '¡Cambio aplicado! ¿Quieres verlo?',
  },
  english: {
    greeting: (brand) => `Hey, I'm the Creative Coach for ${brand}. Ask me anything about weak points and visual improvements.`,
    hint: 'Ask about this creative',
    starters: [
      'What visual elements are missing?',
      'How would you improve it in 3 changes?',
      'Where is the CTA?',
    ],
    thinking: 'thinking...',
    placeholder: 'Ex: where does this creative fail?',
    send: 'Send',
    implement: '⚡ Implement',
    implementing: 'Implementing...',
    implemented: 'Change applied! Want to see it?',
  },
};

const normalizeLanguage = (lang) =>
  ['catalan', 'castilian', 'english'].includes(lang) ? lang : 'catalan';

// A single message bubble — handles implement button for assistant actions
function MessageBubble({ msg, ui, onImplement, implementingId }) {
  const isAssistant = msg.role === 'assistant';
  const hasAction = isAssistant && msg.action?.intent === 'modify';
  const isThisImplementing = implementingId === msg.id;

  return (
    <div className={`flex flex-col ${isAssistant ? 'items-start' : 'items-end'} gap-1.5`}>
      <div className={`max-w-[90%] md:max-w-[75%] rounded-2xl px-4 py-3 text-sm leading-relaxed shadow-sm whitespace-pre-wrap ${
        isAssistant
          ? 'bg-white border border-slate-200 text-slate-700 rounded-bl-md'
          : 'bg-indigo-600 text-white rounded-br-md'
      }`}>
        {msg.content}
      </div>

      {/* Implement CTA — only for modify-intent assistant messages */}
      {hasAction && (
        <button
          onClick={() => onImplement(msg)}
          disabled={!!implementingId}
          className={`flex items-center gap-2 px-4 py-2 rounded-xl text-[11px] font-black uppercase tracking-widest transition-all shadow-lg ${
            isThisImplementing
              ? 'bg-indigo-200 text-indigo-400 cursor-wait'
              : 'bg-indigo-600 hover:bg-indigo-700 text-white shadow-indigo-200 hover:-translate-y-0.5 active:scale-95'
          }`}
        >
          {isThisImplementing
            ? <><span className="w-3 h-3 border-2 border-indigo-400/30 border-t-indigo-500 rounded-full animate-spin" />{ui.implementing}</>
            : ui.implement
          }
        </button>
      )}

      {/* Success state after implement */}
      {msg.implementedUrl && (
        <div className="max-w-[90%] rounded-2xl overflow-hidden border-2 border-emerald-400 shadow-lg shadow-emerald-100">
          <img src={msg.implementedUrl} alt="Implemented change" className="w-full object-contain max-h-48" />
          <p className="text-[10px] font-black uppercase tracking-widest text-emerald-600 text-center py-2 bg-emerald-50">{ui.implemented}</p>
        </div>
      )}
    </div>
  );
}

export default function CreativeChatModal({ creative, isOpen, onClose, onApplyImplement, initialMessages = [], onHistoryChange }) {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isSending, setIsSending] = useState(false);
  const [language, setLanguage] = useState('catalan');
  const [implementingId, setImplementingId] = useState(null);
  const scrollRef = useRef(null);

  const ui = COPY[normalizeLanguage(language)];

  const greeting = useMemo(() => {
    if (!creative) return '';
    const brand = creative.advertiser || creative.advertiser_name || creative.subject || 'AD-EX';
    return ui.greeting(brand);
  }, [creative, ui]);

  // Restore saved history when modal opens; only reset to greeting for brand-new chats
  useEffect(() => {
    if (!isOpen || !creative) return;
    setInput('');
    setIsSending(false);
    setImplementingId(null);
    if (initialMessages && initialMessages.length > 0) {
      setMessages(initialMessages);
    } else {
      setMessages([{
        id: `assistant-${creative.id}-greeting`,
        role: 'assistant',
        content: greeting,
        action: null,
      }]);
    }
  }, [isOpen, creative?.id]);

  // Persist history to parent whenever messages change
  useEffect(() => {
    if (isOpen && onHistoryChange && messages.length > 0) {
      onHistoryChange(messages);
    }
  }, [messages]);

  useEffect(() => {
    if (!scrollRef.current) return;
    scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
  }, [messages, isSending]);

  if (!isOpen || !creative) return null;

  const toApiHistory = (items) =>
    items
      .filter((m) => m.role === 'user' || m.role === 'assistant')
      .map((m) => ({ role: m.role, content: m.content }));

  const sendMessage = async (rawText) => {
    const text = (rawText || '').trim();
    if (!text || isSending) return;

    const userMessage = { id: `user-${Date.now()}`, role: 'user', content: text, action: null };
    const nextMessages = [...messages, userMessage];
    setMessages(nextMessages);
    setInput('');
    setIsSending(true);

    const result = await askCreativeChat(creative.id, text, toApiHistory(nextMessages), language, true /* agentic */);

    const assistantMessage = {
      id: `assistant-${Date.now()}`,
      role: 'assistant',
      content: result.answer,
      action: result.action || null,
    };
    setMessages((prev) => [...prev, assistantMessage]);
    setIsSending(false);
  };

  const handleImplement = async (msg) => {
    if (!msg.action) return;
    if (implementingId) return;  // Already implementing something
    setImplementingId(msg.id);

    try {
      const res = await fetch(`http://localhost:8000/api/creatives/${creative.id}/implement`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          description: msg.action.description,
          diffusion_prompt: msg.action.diffusion_prompt,
        }),
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || `Server error ${res.status}`);
      }

      const data = await res.json();
      if (data.success && data.new_image_url) {
        const urlWithBust = data.new_image_url + '?t=' + Date.now();
        setMessages((prev) =>
          prev.map((m) =>
            m.id === msg.id ? { ...m, implementedUrl: urlWithBust, implementedId: data.creative_id } : m
          )
        );
        if (onApplyImplement) onApplyImplement(data.creative_id || creative.id, data.new_image_url);
      } else {
        throw new Error('No image URL returned from server');
      }
    } catch (e) {
      console.error('[Implement] Failed:', e);
      // Show error message in chat
      setMessages((prev) => [...prev, {
        id: `error-${Date.now()}`,
        role: 'assistant',
        content: `❌ Error implementing: ${e.message}`,
        action: null,
      }]);
    } finally {
      setImplementingId(null);
    }
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    sendMessage(input);
  };

  return (
    <div className="fixed inset-0 z-[110] bg-slate-900/75 backdrop-blur-lg flex items-center justify-center p-4 md:p-8 animate-in fade-in duration-200">
      <div className="w-full max-w-5xl h-[88vh] md:h-[86vh] bg-white rounded-[2rem] md:rounded-[2.5rem] shadow-2xl overflow-hidden border border-white/50 flex flex-col md:flex-row">

        {/* Left sidebar */}
        <aside className="md:w-[300px] bg-gradient-to-b from-slate-900 via-slate-900 to-indigo-950 text-white p-5 md:p-6 border-b md:border-b-0 md:border-r border-white/10 flex flex-col">
          <div className="flex items-start justify-between mb-5">
            <div>
              <p className="text-[9px] uppercase tracking-[0.25em] text-indigo-300 font-black">Creative Coach</p>
              <h3 className="text-xl font-black tracking-tighter italic uppercase mt-1">
                {creative.advertiser || creative.advertiser_name || creative.subject || 'AD-EX'}
              </h3>
            </div>
            <button
              onClick={onClose}
              className="w-8 h-8 rounded-xl bg-white/10 hover:bg-white/20 transition-colors text-sm flex-shrink-0"
              aria-label="Close chat"
            >
              ✕
            </button>
          </div>

          <div className="rounded-2xl overflow-hidden border border-white/10 shadow-xl mb-4 bg-slate-950 flex-shrink-0">
            <img src={creative.image_url} alt={creative.subject} className="w-full aspect-[9/16] object-contain max-h-[35vh]" />
          </div>

          <div className="space-y-1.5 text-[10px] font-bold uppercase tracking-widest text-white/70 flex-shrink-0">
            <div className="flex justify-between"><span>Advertiser</span><span className="text-white">{creative.advertiser}</span></div>
            <div className="flex justify-between"><span>Score</span><span className="text-emerald-300">{creative.performance_score}</span></div>
            <div className="flex justify-between"><span>CTR</span><span className="text-indigo-300">{creative.ctr}%</span></div>
          </div>

          {/* Language selector + image info at bottom */}
          <div className="flex-shrink-0 mt-auto pt-4 border-t border-white/10">
            <p className="text-[9px] uppercase tracking-widest text-white/40 font-black mb-2 text-center">Language</p>
            <div className="flex gap-1.5 justify-center">
              {LANGUAGE_OPTIONS.map((opt) => (
                <button key={opt.value} onClick={() => setLanguage(opt.value)} type="button"
                  className={`px-3 py-1 rounded-lg text-[10px] font-black uppercase tracking-wide transition-colors ${
                    language === opt.value ? 'bg-indigo-600 text-white' : 'bg-white/10 text-white/60 hover:bg-white/20'
                  }`}>
                  {opt.label}
                </button>
              ))}
            </div>
          </div>
        </aside>

        {/* Chat area */}
        <section className="flex-1 flex flex-col min-h-0 bg-slate-50/60">
          {/* Quick starters */}
          <div className="px-5 pt-4 pb-3 border-b border-slate-200 bg-white/80 backdrop-blur-sm">
            <div className="flex flex-wrap gap-1.5">
              {ui.starters.map((q) => (
                <button key={q} onClick={() => sendMessage(q)} disabled={isSending}
                  className="px-3 py-1.5 rounded-full border border-slate-200 bg-white hover:bg-indigo-50 hover:border-indigo-200 text-[11px] font-bold text-slate-600 transition-colors disabled:opacity-50">
                  {q}
                </button>
              ))}
            </div>
          </div>

          {/* Messages */}
          <div ref={scrollRef} className="flex-1 overflow-y-auto p-5 md:p-6 space-y-4">
            {messages.map((m) => (
              <MessageBubble
                key={m.id}
                msg={m}
                ui={ui}
                onImplement={handleImplement}
                implementingId={implementingId}
              />
            ))}

            {isSending && (
              <div className="flex justify-start">
                <div className="bg-white border border-slate-200 text-slate-500 rounded-2xl rounded-bl-md px-4 py-3 text-sm shadow-sm flex items-center gap-2">
                  <span className="w-3 h-3 border-2 border-slate-300 border-t-slate-500 rounded-full animate-spin" />
                  {ui.thinking}
                </div>
              </div>
            )}
          </div>

          {/* Input */}
          <form onSubmit={handleSubmit} className="p-4 border-t border-slate-200 bg-white flex gap-3">
            <input
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder={ui.placeholder}
              className="flex-1 px-4 py-3 rounded-2xl border border-slate-200 focus:outline-none focus:ring-2 focus:ring-indigo-500 text-sm"
              disabled={isSending}
            />
            <button type="submit" disabled={isSending || !input.trim()}
              className="px-5 py-3 rounded-2xl bg-slate-900 hover:bg-indigo-600 text-white font-black text-[10px] uppercase tracking-widest transition-all disabled:opacity-50">
              {ui.send}
            </button>
          </form>
        </section>
      </div>
    </div>
  );
}
