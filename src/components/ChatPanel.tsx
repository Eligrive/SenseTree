import { useEffect, useRef, useState } from "react";
import { Bot, Check, FileText, Loader2, Send, Trash2, Wand2 } from "lucide-react";
import { listen } from "@tauri-apps/api/event";
import type { ActionPlan, ChatSource, ChatTurn, Operation } from "../lib/types";
import { applyActionPlan, chatWithAssistant, discardActionPlan } from "../lib/ipc";
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

/// Rend le texte d'une réponse en linkifiant les noms de fichiers qui correspondent
/// à une source (clic = ouvrir le fichier). Robuste : ne dépend pas du modèle, on
/// détecte les mentions de sources dans le texte produit.
function renderWithCitations(
  text: string,
  sources: ChatSource[] | undefined,
  onOpen: (path: string) => void
) {
  const names = [...new Set((sources ?? []).map((s) => s.name).filter(Boolean))].sort(
    (a, b) => b.length - a.length
  );
  if (names.length === 0) return text;
  const escaped = names.map((n) => n.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));
  const re = new RegExp(`(${escaped.join("|")})`, "g");
  return text.split(re).map((part, i) => {
    const src = sources!.find((s) => s.name === part);
    return src ? (
      <button
        key={i}
        onClick={() => onOpen(src.path)}
        title={src.path}
        className="text-blue-400 underline decoration-dotted underline-offset-2 hover:text-blue-300"
      >
        {part}
      </button>
    ) : (
      <span key={i}>{part}</span>
    );
  });
}

export default function ChatPanel({ currentPath, reasoningOk, onOpenSource }: Props) {
  const [messages, setMessages] = useState<Msg[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  // Trace live des actions de l'agent pendant qu'il travaille (événements backend).
  const [steps, setSteps] = useState<string[]>([]);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [messages, busy, steps.length]);

  const push = (m: Msg) => setMessages((prev) => [...prev, m]);

  /// Vide la conversation. Les plans d'action encore EN ATTENTE sont abandonnés
  /// côté backend, pour ne pas laisser de brouillons orphelins dans le journal.
  const clearChat = () => {
    if (busy) return;
    messages.forEach((m) => {
      if (m.role === "plan" && m.status === "pending" && m.plan.transaction_id != null) {
        discardActionPlan(m.plan.transaction_id).catch(() => {});
      }
    });
    setMessages([]);
    setInput("");
  };

  const send = async (raw?: string) => {
    const text = (raw ?? input).trim();
    if (!text || busy) return;
    setInput("");
    push({ id: nextId(), role: "user", text });
    setBusy(true);
    setSteps([]);

    // Écoute les étapes de l'agent (recherche, lecture, outil…) émises en direct.
    let unlisten: (() => void) | undefined;
    try {
      unlisten = await listen<{ label: string }>("agent-step", (e) => {
        setSteps((prev) => [...prev, e.payload.label]);
      });

      const history: ChatTurn[] = messages
        .filter((m): m is Extract<Msg, { role: "user" | "assistant" }> => m.role !== "plan")
        .map((m) => ({ role: m.role, content: m.text }));
      history.push({ role: "user", content: text });
      const res = await chatWithAssistant(history, currentPath ?? undefined);
      if (res.plan) {
        push({ id: nextId(), role: "plan", plan: res.plan, status: "pending" });
      } else {
        push({ id: nextId(), role: "assistant", text: res.answer ?? "", sources: res.sources });
      }
    } catch (e) {
      push({ id: nextId(), role: "assistant", text: `⚠️ ${String(e)}` });
    } finally {
      unlisten?.();
      setSteps([]);
      setBusy(false);
    }
  };

  const approve = async (msgId: number, plan: ActionPlan, operations: Operation[]) => {
    if (plan.transaction_id == null) return;
    try {
      const res = await applyActionPlan(plan.transaction_id, operations);
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
    <aside className="flex h-full w-full flex-col bg-zinc-950/60">
      <div className="flex items-center gap-2 border-b border-zinc-800 px-4 py-3.5">
        <Bot size={17} className="text-blue-400" />
        <span className="text-sm font-semibold text-zinc-200">Assistant</span>
        <span className="ml-auto truncate text-[11px] text-zinc-500" title={currentPath ?? ""}>
          {currentPath ? currentPath.split(/[\\/]/).pop() : "—"}
        </span>
        {messages.length > 0 && (
          <button
            onClick={clearChat}
            disabled={busy}
            title="Vider la conversation (les plans en attente sont abandonnés)"
            className="shrink-0 rounded p-1 text-zinc-500 transition hover:bg-zinc-800 hover:text-zinc-200 disabled:opacity-40"
          >
            <Trash2 size={15} />
          </button>
        )}
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
              onApprove={(ops) => approve(m.id, m.plan, ops)}
              onDiscard={() => discard(m.id, m.plan)}
            />
          ) : (
            <div key={m.id} className={m.role === "user" ? "ml-auto max-w-[90%]" : "max-w-[95%]"}>
              <div
                className={`rounded-2xl px-3.5 py-2 text-sm ${
                  m.role === "user" ? "bg-blue-600 text-white" : "bg-zinc-800/80 text-zinc-200"
                }`}
              >
                <p className="whitespace-pre-wrap break-words">
                  {m.role === "assistant"
                    ? renderWithCitations(m.text, m.sources, onOpenSource)
                    : m.text}
                </p>
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
          <div className="space-y-1.5">
            {steps.length === 0 ? (
              <div className="flex items-center gap-2 text-xs text-zinc-500">
                <Loader2 size={13} className="animate-spin" /> réflexion…
              </div>
            ) : (
              steps.map((s, i) => {
                const last = i === steps.length - 1;
                return (
                  <div
                    key={i}
                    className={`flex items-center gap-2 text-xs ${
                      last ? "text-zinc-300" : "text-zinc-500"
                    }`}
                  >
                    {last ? (
                      <Loader2 size={12} className="shrink-0 animate-spin" />
                    ) : (
                      <Check size={12} className="shrink-0 text-emerald-500" />
                    )}
                    <span className="truncate">{s}</span>
                  </div>
                );
              })
            )}
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
