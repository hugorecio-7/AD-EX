import React, { useEffect, useMemo, useRef, useState } from 'react';
import { askCreativeChat } from '../services/api';

const LANGUAGE_OPTIONS = [
  { value: 'catalan', label: 'Catala' },
  { value: 'castilian', label: 'Castellano' },
  { value: 'english', label: 'English' },
];

const COPY = {
  catalan: {
    greeting: (subject) => `Ei, soc el Creative Coach de ${subject}. Pregunta'm el que vulguis sobre punts febles, millores visuals i decisions de disseny.`,
    hint: 'Pregunta coses concretes del creative i com millorar-lo',
    starters: [
      'Que li falta visualment a aquest anunci?',
      'Com el milloraries en 3 canvis concrets?',
      'On esta el CTA i es prou visible?',
    ],
    thinking: 'pensant una resposta...',
    placeholder: 'Ex: on falla aquest creative i com el milloro?',
    send: 'enviar',
  },
  castilian: {
    greeting: (subject) => `Hola, soy el Creative Coach de ${subject}. Preguntame lo que quieras sobre puntos debiles, mejoras visuales y decisiones de diseno.`,
    hint: 'Pregunta cosas concretas del creative y como mejorarlo',
    starters: [
      'Que le falta visualmente a este anuncio?',
      'Como lo mejorarias en 3 cambios concretos?',
      'Donde esta el CTA y se ve bien?',
    ],
    thinking: 'pensando una respuesta...',
    placeholder: 'Ej: donde falla este creative y como lo mejoro?',
    send: 'enviar',
  },
  english: {
    greeting: (subject) => `Hey, I am the Creative Coach for ${subject}. Ask me anything about weak points, visual improvements, and design decisions.`,
    hint: 'Ask focused questions about this creative and how to improve it',
    starters: [
      'What visual elements are missing in this ad?',
      'How would you improve it in 3 concrete changes?',
      'Where is the CTA and is it visible enough?',
    ],
    thinking: 'thinking of a response...',
    placeholder: 'Ex: where does this creative fail and how can I improve it?',
    send: 'send',
  },
};

const normalizeLanguage = (lang) => (['catalan', 'castilian', 'english'].includes(lang) ? lang : 'catalan');

export default function CreativeChatModal({ creative, isOpen, onClose }) {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isSending, setIsSending] = useState(false);
  const [language, setLanguage] = useState('catalan');
  const scrollRef = useRef(null);

  const ui = COPY[normalizeLanguage(language)];

  const greeting = useMemo(() => {
    if (!creative) return '';
    const brand = creative.advertiser || creative.advertiser_name || creative.subject || 'Smadex';
    return ui.greeting(brand);
  }, [creative, ui]);

  useEffect(() => {
    if (!isOpen || !creative) return;
    setInput('');
    setIsSending(false);
    setMessages([
      {
        id: `assistant-${creative.id}-greeting`,
        role: 'assistant',
        content: greeting,
      },
    ]);
  }, [isOpen, creative, greeting]);

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

    const userMessage = {
      id: `user-${Date.now()}`,
      role: 'user',
      content: text,
    };

    const nextMessages = [...messages, userMessage];
    setMessages(nextMessages);
    setInput('');
    setIsSending(true);

    const result = await askCreativeChat(creative.id, text, toApiHistory(nextMessages), language);

    const assistantMessage = {
      id: `assistant-${Date.now()}`,
      role: 'assistant',
      content: result.answer,
    };

    setMessages((prev) => [...prev, assistantMessage]);
    setIsSending(false);
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    sendMessage(input);
  };

  return (
    <div className="fixed inset-0 z-[110] bg-slate-900/75 backdrop-blur-lg flex items-center justify-center p-4 md:p-8 animate-in fade-in duration-200">
      <div className="w-full max-w-5xl h-[88vh] md:h-[86vh] bg-white rounded-[2rem] md:rounded-[2.5rem] shadow-2xl overflow-hidden border border-white/50 flex flex-col md:flex-row">
        <aside className="md:w-[340px] bg-gradient-to-b from-slate-900 via-slate-900 to-indigo-950 text-white p-6 md:p-8 border-b md:border-b-0 md:border-r border-white/10">
          <div className="flex items-start justify-between mb-6">
            <div>
              <p className="text-[10px] uppercase tracking-[0.25em] text-indigo-300 font-black">Creative Coach</p>
              <h3 className="text-2xl font-black tracking-tighter italic uppercase mt-2">
                {creative.advertiser || creative.advertiser_name || creative.subject || 'Smadex'}
              </h3>
            </div>
            <button
              onClick={onClose}
              className="w-10 h-10 rounded-xl bg-white/10 hover:bg-white/20 transition-colors text-lg"
              aria-label="Close chat"
            >
              ✕
            </button>
          </div>

          <div className="rounded-2xl overflow-hidden border border-white/10 shadow-xl mb-5 bg-slate-950">
            <img src={creative.image_url} alt={creative.subject} className="w-full aspect-[9/16] object-contain" />
          </div>

          <div className="space-y-2 text-[11px] font-bold uppercase tracking-widest text-white/70">
            <div className="flex justify-between"><span>Advertiser</span><span className="text-white">{creative.advertiser}</span></div>
            <div className="flex justify-between"><span>Score</span><span className="text-emerald-300">{creative.performance_score}</span></div>
            <div className="flex justify-between"><span>CTR</span><span className="text-indigo-300">{creative.ctr}%</span></div>
          </div>
        </aside>

        <section className="flex-1 flex flex-col min-h-0 bg-slate-50/60">
          <div className="px-5 md:px-8 pt-5 md:pt-7 pb-4 border-b border-slate-200 bg-white/80 backdrop-blur-sm">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <p className="text-[10px] md:text-xs text-slate-500 font-black uppercase tracking-[0.18em]">{ui.hint}</p>
              <div className="bg-slate-100 rounded-xl p-1 flex gap-1">
                {LANGUAGE_OPTIONS.map((opt) => (
                  <button
                    key={opt.value}
                    onClick={() => setLanguage(opt.value)}
                    className={`px-2.5 py-1 rounded-lg text-[10px] font-black uppercase tracking-wide transition-colors ${language === opt.value
                        ? 'bg-indigo-600 text-white shadow'
                        : 'text-slate-500 hover:bg-white'
                      }`}
                    disabled={isSending}
                    type="button"
                  >
                    {opt.label}
                  </button>
                ))}
              </div>
            </div>
            <div className="mt-3 flex flex-wrap gap-2">
              {ui.starters.map((q) => (
                <button
                  key={q}
                  onClick={() => sendMessage(q)}
                  disabled={isSending}
                  className="px-3 py-1.5 rounded-full border border-slate-200 bg-white hover:bg-indigo-50 hover:border-indigo-200 text-[11px] font-bold text-slate-600 transition-colors disabled:opacity-50"
                >
                  {q}
                </button>
              ))}
            </div>
          </div>

          <div ref={scrollRef} className="flex-1 overflow-y-auto p-5 md:p-8 space-y-4">
            {messages.map((m) => (
              <div key={m.id} className={`flex ${m.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                <div
                  className={`max-w-[90%] md:max-w-[75%] rounded-2xl px-4 py-3 text-sm leading-relaxed shadow-sm ${m.role === 'user'
                      ? 'bg-indigo-600 text-white rounded-br-md'
                      : 'bg-white border border-slate-200 text-slate-700 rounded-bl-md'
                    }`}
                >
                  {m.content}
                </div>
              </div>
            ))}

            {isSending && (
              <div className="flex justify-start">
                <div className="bg-white border border-slate-200 text-slate-500 rounded-2xl rounded-bl-md px-4 py-3 text-sm shadow-sm">
                  {ui.thinking}
                </div>
              </div>
            )}
          </div>

          <form onSubmit={handleSubmit} className="p-4 md:p-6 border-t border-slate-200 bg-white flex gap-3">
            <input
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder={ui.placeholder}
              className="flex-1 px-4 py-3 rounded-2xl border border-slate-200 focus:outline-none focus:ring-2 focus:ring-indigo-500 text-sm"
              disabled={isSending}
            />
            <button
              type="submit"
              disabled={isSending || !input.trim()}
              className="px-5 md:px-6 py-3 rounded-2xl bg-slate-900 hover:bg-indigo-600 text-white font-black text-[10px] md:text-xs uppercase tracking-widest transition-all disabled:opacity-50"
            >
              {ui.send}
            </button>
          </form>
        </section>
      </div>
    </div>
  );
}
