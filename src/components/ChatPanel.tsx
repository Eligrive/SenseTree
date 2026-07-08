import { useEffect, useRef, useState } from "react";
import { Bot, FileText, Loader2, Send, Wand2 } from "lucide-react";
import type { ActionPlan, ChatSource, ChatTurn } from "../lib/types";
import {
  applyActionPlan,
  chatWithAssistant,
  discardActionPlan,
  planReorganization,
} from "../lib/ipc";
import ActionPlanCard from "./ActionPlanCard";

interface Props {
  currentPath: string | null;
  reasoningOk: boolean;
  onOpenSource: (path: string) => void;
}

type Msg =
  | { id: number; role: "user" | "assistant"; text: string; sources?: ChatSource[] }
  | { id: number; role: "plan"; plan: ActionPlan; status: "pending" | "applied" | "discarded" };

let uid = 0;
const nextId = () => ++uid;

// Une instruction d'action (vs une simple question) déclenche un plan Dry-Run.
const ACTION_RE = /(réorganis|reorganis|range|rang\b|trier|tri\b|classe|renomm|nettoie|nettoy|supprim|déplace|deplace|organis)/i;

export default function ChatPanel({ currentPath, reasoningOk, onOpenSource }: Props) {
  const [messages, setMessages] = useState<Msg[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [messages, busy]);

  const push = (m: Msg) => setMessages((prev) => [...prev, m]);

  const send = async (raw?: string) => {
    const text = (raw ?? input).trim();
    if (!text || busy) return;
    setInput("");
    push({ id: nextId(), role: "user", text });
    setBusy(true);

    try {
      if (ACTION_RE.test(text) && currentPath) {
        const plan = await planReorganization(text, currentPath);
        push({ id: nextId(), role: "plan", plan, status: "pending" });
      } else {
        const history: ChatTurn[] = messages
          .filter((m): m is Extract<Msg, { role: "user" | "assistant" }> => m.role !== "plan")
          .map((m) => ({ role: m.role, content: m.text }));
        history.push({ role: "user", content: text });
        const res = await chatWithAssistant(history, currentPath ?? undefined);
        push({ id: nextId(), role: "assistant", text: res.answer, sources: res.sources });
      }
    } catch (e) {
      push({ id: nextId(), role: "assistant", text: `⚠️ ${String(e)}` });
    } finally {
      setBusy(false);
    }
  };

  const approve = async (msgId: number, plan: ActionPlan) => {
    if (plan.transaction_id == null) return;
    try {
      const res = await applyActionPlan(plan.transaction_id);
      setStatus(msgId, "applied");
      push({ id: nextId(), role: "assistant", text: `✅ ${res.message}` });
    } catch (e) {
      push({ id: nextId(), role: "assistant", text: `⚠️ ${String(e)}` });
    }
  };

  const discard = async (msgId: number, plan: ActionPlan) => {
    if (plan.transaction_id != null) await discardActionPlan(plan.transaction_id).catch(() => {});
    setStatus(msgId, "discarded");
  };

  const setStatus = (msgId: number, status: "applied" | "discarded") =>
    setMessages((prev) =>
      prev.map((m) => (m.id === msgId && m.role === "plan" ? { ...m, status } : m))
    );

  return (
    <aside className="flex h-full w-96 flex-col border-l border-zinc-800 bg-zinc-950/60">
      <div className="flex items-center gap-2 border-b border-zinc-800 px-4 py-3.5">
        <Bot size={17} className="text-blue-400" />
        <span className="text-sm font-semibold text-zinc-200">Assistant</span>
        <span className="ml-auto truncate text-[11px] text-zinc-500" title={currentPath ?? ""}>
          {currentPath ? currentPath.split(/[\\/]/).pop() : "—"}
        </span>
      </div>

      <div ref={scrollRef} className="flex-1 space-y-3 overflow-y-auto p-4">
        {messages.length === 0 && (
          <div className="mt-6 text-center text-sm text-zinc-500">
            <Wand2 className="mx-auto mb-2 text-zinc-600" />
            Posez une question sur vos fichiers,
            <br /> ou demandez : « range ce dossier ».
          </div>
        )}

        {messages.map((m) =>
          m.role === "plan" ? (
            <ActionPlanCard
              key={m.id}
              plan={m.plan}
              status={m.status}
              onApprove={() => approve(m.id, m.plan)}
              onDiscard={() => discard(m.id, m.plan)}
            />
          ) : (
            <div key={m.id} className={m.role === "user" ? "ml-auto max-w-[90%]" : "max-w-[95%]"}>
              <div
                className={`rounded-2xl px-3.5 py-2 text-sm ${
                  m.role === "user" ? "bg-blue-600 text-white" : "bg-zinc-800/80 text-zinc-200"
                }`}
              >
                <p className="whitespace-pre-wrap break-words">{m.text}</p>
              </div>
              {m.role === "assistant" && m.sources && m.sources.length > 0 && (
                <div className="mt-1.5 flex flex-wrap gap-1.5">
                  {m.sources.map((s) => (
                    <button
                      key={s.path}
                      onClick={() => onOpenSource(s.path)}
                      title={s.path}
                      className="flex max-w-[95%] items-center gap-1 rounded-md bg-zinc-800/60 px-1.5 py-0.5 text-[11px] text-zinc-300 transition hover:bg-zinc-700"
                    >
                      <FileText size={11} className="shrink-0 text-emerald-400" />
                      <span className="truncate">{s.name}</span>
                    </button>
                  ))}
                </div>
              )}
            </div>
          )
        )}

        {busy && (
          <div className="flex items-center gap-2 text-xs text-zinc-500">
            <Loader2 size={13} className="animate-spin" /> réflexion…
          </div>
        )}
      </div>

      <div className="border-t border-zinc-800 p-3">
        {!reasoningOk && (
          <p className="mb-2 rounded-md bg-amber-500/10 px-2 py-1 text-[11px] text-amber-400">
            Aucun modèle de reasoning détecté — configurez-en un dans les Paramètres.
          </p>
        )}
        <div className="flex items-end gap-2">
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                send();
              }
            }}
            rows={1}
            disabled={!reasoningOk || busy}
            placeholder={reasoningOk ? "Écrire un message…" : "Reasoning indisponible"}
            className="max-h-32 flex-1 resize-none rounded-xl border border-zinc-800 bg-zinc-900 px-3 py-2 text-sm text-zinc-200 placeholder-zinc-500 outline-none focus:border-blue-500 disabled:opacity-50"
          />
          <button
            onClick={() => send()}
            disabled={!reasoningOk || busy || !input.trim()}
            className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-blue-600 text-white hover:bg-blue-500 disabled:opacity-40"
          >
            <Send size={15} />
          </button>
        </div>
      </div>
    </aside>
  );
}
