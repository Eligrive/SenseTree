import { useEffect, useState } from "react";
import { Gauge, RotateCcw } from "lucide-react";
import { indexingThroughput, resetThroughput } from "../lib/ipc";
import type { StageStats, Throughput } from "../lib/types";

/// Débit des trois étages IA.
///
/// Ce panneau mesure la vitesse des MODÈLES — le temps passé dans chaque étage — et
/// non l'avancement de l'indexation, qui se lit sur la barre de progression. Les deux
/// sont utiles et différents : un pipeline lent avec des modèles rapides signifie que
/// le goulot est ailleurs (lecture disque, extraction PDF, écriture vectorielle).
///
/// L'unité affichée dépend de l'étage, parce qu'elle n'a pas le même sens partout :
///   * vision et reasoning font un appel par fichier → `fichiers/s` et `ms/appel` ;
///   * l'embedding traite N chunks proportionnels au volume → `chunks/s` et `Mo/s`,
///     un `fichiers/s` y serait trompeur.
const LABELS: Record<string, { titre: string; unite: string }> = {
  vision: { titre: "Vision", unite: "img/s" },
  media: { titre: "Média", unite: "médias/s" },
  reasoning: { titre: "Reasoning", unite: "appels/s" },
  embedding: { titre: "Embedding", unite: "chunks/s" },
};

const num = (x: number, d = 1) =>
  x.toLocaleString("fr-FR", { minimumFractionDigits: d, maximumFractionDigits: d });

function Ligne({ s, wall }: { s: StageStats; wall: number }) {
  const meta = LABELS[s.stage] ?? { titre: s.stage, unite: "/s" };
  // Aucun appel : l'étage est inactif (désactivé, ou aucun fichier concerné). On le
  // montre quand même, en grisé — une ligne absente ferait croire à un bug.
  if (s.ops === 0) {
    return (
      <div className="flex items-baseline justify-between text-[11px] text-zinc-600">
        <span>{meta.titre}</span>
        <span>{s.errors > 0 ? `${s.errors} échec(s)` : "inactif"}</span>
      </div>
    );
  }
  // Part du temps réel passée dans cet étage : c'est ce rapport qui désigne le goulot.
  const part = wall > 0 ? Math.min(100, Math.round((s.seconds / wall) * 100)) : null;
  return (
    <div className="space-y-0.5">
      <div className="flex items-baseline justify-between text-[11px]">
        <span className="text-zinc-300">{meta.titre}</span>
        <span className="font-mono text-zinc-400">
          {s.ms_per_op != null ? `${num(s.ms_per_op, 0)} ms/appel` : "—"}
        </span>
      </div>
      <div className="flex items-baseline justify-between text-[10px] text-zinc-500">
        <span>
          {s.units_per_sec != null ? `${num(s.units_per_sec, 2)} ${meta.unite}` : "—"}
          {s.stage === "embedding" && s.mb_per_sec != null && ` · ${num(s.mb_per_sec, 2)} Mo/s`}
        </span>
        <span title="Part du temps écoulé passée dans cet étage">
          {s.ops} appels{part != null ? ` · ${part}%` : ""}
        </span>
      </div>
      {s.errors > 0 && (
        <div className="text-[10px] text-amber-400/80">
          {s.errors} échec(s) — non comptés dans la moyenne
        </div>
      )}
    </div>
  );
}

export default function ThroughputPanel() {
  const [t, setT] = useState<Throughput | null>(null);

  useEffect(() => {
    let vivant = true;
    const lire = () =>
      indexingThroughput()
        .then((x) => vivant && setT(x))
        .catch(() => {});
    lire();
    const id = setInterval(lire, 2000);
    return () => {
      vivant = false;
      clearInterval(id);
    };
  }, []);

  // Rien mesuré du tout : inutile d'occuper la place.
  if (!t || t.stages.every((s) => s.ops === 0 && s.errors === 0)) return null;

  return (
    <div className="space-y-2 rounded-md bg-zinc-900/60 p-2.5">
      <div className="flex items-center justify-between text-[10px] font-semibold uppercase tracking-widest text-zinc-500">
        <span className="flex items-center gap-1.5">
          <Gauge size={11} /> Débit des modèles
        </span>
        <button
          onClick={() => resetThroughput().then(() => indexingThroughput().then(setT))}
          title="Remettre les compteurs à zéro pour chronométrer une indexation précise"
          className="text-zinc-500 hover:text-zinc-300"
        >
          <RotateCcw size={11} />
        </button>
      </div>
      {t.stages.map((s) => (
        <Ligne key={s.stage} s={s} wall={t.wall_seconds} />
      ))}
    </div>
  );
}
