# Guide d'utilisation — SenseTree

SenseTree est un **explorateur de fichiers augmenté**, local-first. Il ne remplace
pas l'explorateur de Windows : il se superpose à ton système de fichiers réel et
y ajoute une **couche de compréhension sémantique** (recherche par le sens, agent
IA sur tes fichiers, rangement assisté). Tout tourne **en local** : tes données ne
quittent jamais ta machine, sauf vers les serveurs IA que tu configures toi-même.

---

## 1. Ce que fait l'application

| Capacité | Description |
|---|---|
| **Indexation sémantique** | Chaque fichier reçoit un **sens** (2 à 4 phrases disant ce que c'est) et un ou plusieurs vecteurs, calculés à partir de son **contenu** (texte, PDF, DOCX, image via vision, audio/vidéo via transcription) ou, à défaut, de son **contexte** (nom + dossier + voisinage). |
| **Recherche hybride** | Le **sens** (vecteurs) *et* les **mots exacts** (BM25) sont cherchés en parallèle, fusionnés, puis réordonnés par un modèle de reranking local. « fichiers liés à mon voyage en Corée » comme « FR7630004000031234567890143 » fonctionnent. |
| **Recherche d'images** | Retrouve des photos par ce qu'elles **montrent** (CLIP), sans légende ni serveur de vision. |
| **Chat agentique** | L'assistant **cherche, lit, liste, recoupe** avant de répondre, cite ses sources, et se souvient de ce que tu lui confies d'une conversation à l'autre. |
| **Rangement assisté (Dry-Run)** | L'IA propose déplacements / renommages / suppressions / corrections de sens. **Rien n'est appliqué sans ta validation explicite.** |
| **Diagnostic « gardener »** | Détecte doublons exacts, dossiers vides, arborescences trop profondes, dossiers fourre-tout. |

L'application **ne modifie jamais tes fichiers silencieusement**. Toute action sur
le disque passe par un aperçu « Avant / Après » que tu dois approuver.

---

## 2. Comment ça marche (sous le capot)

```
                 ┌─────────────┐     ┌──────────────┐
Disque ──▶ Crawler / Watchdog ─▶ File d'attente ─▶ Worker ─▶ Embedding ─▶ LanceDB
                 └─────────────┘     (SQLite)       │        (vecteurs)   (recherche)
                                                     ▼
                                    Texte / Vision / Média / Contexte
```

1. **Crawler** — au démarrage, parcourt tes dossiers racines et met les fichiers
   nouveaux ou modifiés dans une file d'attente (SQLite).
2. **Watchdog** — surveille en temps réel les créations / modifications /
   suppressions et alimente la même file.
3. **Worker** (tâche de fond) — pour chaque fichier :
   extraction du contenu → empreinte SHA-256 (pour ne pas ré-indexer l'identique)
   → étages IA (vision / média / qualification) → découpage en morceaux →
   **embedding** → stockage dans **LanceDB**.
4. **Recherche** — ta requête est vectorisée *et* cherchée en mots-clés ; les deux
   classements sont fusionnés, le haut du panier est réordonné par un
   cross-encoder, et les fichiers dont l'existence est confirmée sur le disque
   sont renvoyés, un seul résultat par fichier.

Quatre voies d'« extraction du sens » selon le fichier :

- **Textuelle** : PDF, DOCX, PPTX, XLSX, HTML, code, `.txt`, Markdown… → contenu réel.
- **Visuelle** : images, et PDF scannés dont aucun texte n'est extractible —
  leurs pages sont alors dessinées puis soumises à un modèle de vision, qui
  en donne une description et en transcrit le texte (si activé).
- **Média** : fichiers audio et vidéo, **tous formats et toutes tailles**. Ils
  sont envoyés tels quels au serveur que tu as configuré, qui accepte ou refuse ;
  un refus fait simplement retomber le fichier sur son contexte. Deux sources de
  sens, activables séparément et combinées : la **transcription** de la parole,
  et la **description visuelle** de l'image pour les vidéos (ce qu'on y voit, en
  plus de ce qui s'y dit). Le résultat est découpé et indexé comme un document,
  ce qui rend l'enregistrement cherchable sur n'importe lequel de ses passages.
- **Contextuelle** : fichiers illisibles (machines virtuelles, binaires, gros
  fichiers) → indexés par leur nom, dossier, voisinage et type, enrichis d'une
  **devinette** du modèle sur leur nature probable. Ils restent donc trouvables.

Les dossiers techniques (`venv`, `node_modules`, bundles d'application, packs de
samples DAW…) sont détectés et indexés **en bloc** — une seule description pour
tout le dossier, sans descendre dedans — pour garder l'index propre sans que tu
aies à faire le tri.

### Où sont stockées les données

Tout est local, dans `…/AppData/Roaming/com.virgi.sensetree/` :

- `sensetree.sqlite` — catalogue, file d'indexation, sens extraits, profils de
  dossiers, mémoire de l'agent, journal des actions.
- `lancedb/` — base vectorielle (les embeddings, et l'index BM25).
- `settings.json` — ta configuration (modèles, serveurs, prompts, dossiers).
- `models/` — modèles locaux téléchargés (embedding, reranker, CLIP) et le runtime ONNX.
- `trash/` — corbeille interne (les « suppressions » y sont déplacées, donc réversibles).

---

## 3. Premier lancement

Au tout premier démarrage :

1. Le **modèle d'embedding local** (~130 Mo) et le **runtime ONNX** se téléchargent
   une fois, puis sont mis en cache.
2. L'**indexation démarre** en tâche de fond sur ton dossier Documents (dossier
   racine par défaut, modifiable dans les Paramètres).
3. Tu peux utiliser l'app immédiatement ; la recherche devient de plus en plus
   complète à mesure que les fichiers sont indexés.

> ⏳ L'indexation initiale peut être longue (plusieurs milliers de fichiers en
> CPU). Suis l'avancement grâce à l'**indicateur d'indexation** dans la barre de
> gauche, ou ouvre la **file d'indexation** pour voir le détail fichier par fichier.

L'application **se met à jour toute seule** : au démarrage, elle vérifie s'il
existe une version plus récente (signée cryptographiquement) et te propose de
l'installer. Rien ne s'installe sans ton clic.

---

## 4. L'interface

L'écran est divisé en **trois panneaux**.

### 4.1. Barre latérale (gauche)

- **Dossiers indexés** — tes racines. Clique pour naviguer dedans. Le dossier
  actif est surligné en bleu. Une pastille de santé signale les dossiers
  encombrés, les doublons ou les dossiers vides.
- **Indicateur d'indexation** — barre de progression + compteur :
  - `X à indexer` (en ambre) et `Y docs` (total en file).
  - Barre **bleue** avec spinner = indexation en cours ; **verte** avec ✓ = à jour.
  - Une mention rouge « N en échec » apparaît si des fichiers n'ont pas pu être traités.
  - Bouton **pause / reprise** : gèle la file et libère le modèle local.
  - Clic sur le compteur → ouvre la **file d'indexation** en détail (fichier en
    cours et étages IA qu'il traverse, en attente, en échec, avec relance ou mise
    à l'écart).
  - Une mention « classification reportée » signale des dossiers en attente d'une
    décision IA : ils repartent tout seuls dès que le modèle de reasoning répond.
- **Diagnostic du dossier** — lance l'analyse « gardener ».
- **Recherche d'images** — recherche par similarité visuelle (CLIP).
- **État IA** — un voyant par créneau : `Embedding` (indexation), `Reasoning`
  (chat), `Vision` (images), `Transcription` et `Description vidéo`. 🟢 vert =
  prêt · ⚫ gris = indisponible ou désactivé. Survole pour le détail. Cinq
  voyants distincts plutôt qu'un global : c'est ce qui permet de voir *lequel*
  des serveurs ne répond pas.
  Un voyant média affichant « connecté (pas d'inventaire /models) » est normal —
  plusieurs serveurs de transcription, whisper.cpp en tête, n'exposent pas cette
  route ; leur 404 prouve quand même qu'ils répondent.
- **Paramètres** — configuration complète.

### 4.2. Explorateur (centre)

- **Barre de recherche sémantique** (en haut) — tape une requête en langage
  naturel et valide. Sous la barre, une case **« Limiter au dossier courant »** :
  - décochée (défaut) = recherche **globale** sur tout l'index ;
  - cochée = recherche restreinte au dossier ouvert.
- **Fil d'ariane** — chemin cliquable pour remonter dans l'arborescence.
- **Bascule Liste / Arbre** — l'**arbre** affiche le dossier sous forme de carte
  de pertinence : chaque branche est colorée selon sa proximité avec ta requête.
- **Liste de fichiers** — colonnes Nom / Taille / Modifié.
  - Icône colorée selon le type (dossier, image, document, texte/code…).
  - **Pastille de statut d'indexation** à droite du nom :
    🟢 indexé · 🟡 en attente · 🔴 échec · (aucune) = non concerné.
  - **Double-clic** : ouvre le dossier, ou ouvre le fichier avec l'application par défaut.
  - **Clic simple** : ouvre le **panneau de détail** — sens extrait, contenu
    extrait, statut, actions (re-qualifier, ré-indexer, discuter de ce fichier).
- **Vue résultats de recherche** — remplace la liste quand une recherche est
  active. Chaque résultat affiche le nom, le chemin, un **score de pertinence
  (%)** et un extrait.

### 4.3. Assistant / Chat (droite)

- Zone de conversation + champ de saisie (Entrée pour envoyer, Maj+Entrée pour
  aller à la ligne). Le chemin du dossier courant est affiché en en-tête : c'est
  la portée de la conversation.
- L'assistant est un **agent** : il utilise des outils avant de répondre, et tu
  vois chaque action en direct (`🔍 Recherche…`, `📄 Lecture…`, `📂 Exploration…`).
  Il dispose de six outils intégrés — chercher, lire un fichier, lister un
  dossier, lire les sens extraits, proposer un plan d'action, mémoriser — plus
  ceux des serveurs MCP que tu ajoutes.
- **Deux issues possibles** :
  - **Question** (« résume les documents de ce dossier », « y a-t-il un contrat
    ici ? ») → réponse en texte, avec les **fichiers cités cliquables**.
  - **Instruction d'action** (« range ce dossier », « renomme ces factures »,
    « corrige les descriptions fausses ») → génère un **plan Dry-Run** (voir §6).
- Si aucun modèle de reasoning n'est configuré, le champ est désactivé.

---

## 5. La recherche

- Formule ta requête **par le sens** : SenseTree comprend le thème, pas seulement
  les mots exacts. « photos de montagne » peut remonter un fichier `IMG_4821.jpg`
  décrit par la vision comme « paysage alpin enneigé ».
- Mais les **mots exacts marchent aussi** : numéro de série, IBAN, nom propre,
  extension. C'est la moitié BM25 de la recherche hybride qui s'en charge.
- **Portée** : globale par défaut ; coche « Limiter au dossier courant » pour
  cibler une branche précise du disque.
- **Score** : pourcentage de pertinence, comparable **à l'intérieur d'une même
  liste** de résultats. Un seul résultat par fichier (le meilleur passage).
- Un résultat n'apparaît que si le fichier **existe toujours** physiquement.
- **Recherche d'images** (barre latérale) : décris ce que tu cherches à voir
  (« coucher de soleil sur un lac »). Clique d'abord sur **« Indexer les images »**
  — l'index visuel est construit à la demande, et n'est pas mis à jour tout seul.

> Si une recherche ne renvoie rien : le fichier n'est peut-être **pas encore
> indexé**, il est dans un **dossier-bloc**, ou ta portée est restreinte.

---

## 6. Chat & rangement assisté (Dry-Run)

Quand tu demandes une réorganisation, l'assistant renvoie une **carte de plan
d'action** au lieu d'agir directement :

- En-tête « **Plan d'action · Dry-Run** » + nombre d'opérations.
- Chaque ligne montre l'opération et **la raison donnée par l'IA** :
  - `ancien nom → nouveau nom` (déplacement / renommage) ;
  - `nom (barré) → corbeille` (suppression) ;
  - `nom` avec icône dossier (création de dossier) ;
  - `sens corrigé` (re-qualification : le **fichier n'est pas touché**, seule sa
    description change).
- **Coche / décoche** les opérations que tu veux garder avant d'appliquer.
- Deux boutons :
  - **Appliquer** — exécute réellement les opérations, de façon
    **transactionnelle** : si une opération échoue (fichier verrouillé…),
    **tout est annulé** (rollback) pour ne jamais laisser l'arborescence à moitié
    modifiée. L'index vectoriel est mis à jour sans re-calcul (les fichiers
    déplacés gardent leur sens).
  - **Annuler** — abandonne le plan, rien n'est touché.
- Les **suppressions** ne sont pas destructives : les fichiers sont déplacés dans
  la corbeille interne (`trash/`), donc récupérables.
- Sécurité : un plan qui sortirait de tes dossiers racines configurés est refusé,
  et seules les opérations réellement présentes dans le plan validé sont acceptées.

**Corriger un sens faux.** Si la description d'un fichier est erronée, tu peux la
réécrire depuis le panneau de détail, ou demander à l'agent de le faire (« les
descriptions de ce dossier sont fausses, corrige-les ») — il lit les sens
existants et te propose un plan de corrections. Un sens corrigé est **épinglé** :
aucune ré-indexation ne le régénère, et le fichier est ré-embeddé pour que la
correction soit effective **dans la recherche**, pas seulement à l'écran.

---

## 7. Diagnostic « gardener »

Bouton **« Diagnostic du dossier »** (barre latérale) → analyse le dossier courant
et affiche :

- Nombre de fichiers, profondeur maximale, nombre de groupes de doublons.
- **Suggestions** : dossier fourre-tout (plus de 40 fichiers en vrac),
  arborescence trop profonde (plus de 6 niveaux), dossiers vides, doublons exacts
  (même contenu, détecté par empreinte SHA-256).
- La liste des doublons et des dossiers vides.

Un audit de fond tourne aussi périodiquement et alimente les **pastilles de santé**
des racines dans la barre latérale.

Le diagnostic est **en lecture seule** : il ne fait que constater. Pour agir,
passe par le chat (Dry-Run).

---

## 8. Paramètres

SenseTree se connecte au(x) moteur(s) IA de **ton choix**. Tous les endpoints
suivent le standard **compatible OpenAI** (`/v1/...`) : Ollama, LM Studio, vLLM,
un serveur maison sur ton réseau, ou une API externe.

### Embedding (indexation)

- **Mode Local** (défaut) : modèle ONNX embarqué (`multilingual-e5-small` par
  défaut). Douze modèles proposés, avec leur dimension **et surtout leur caractère
  multilingue** — invisible dans le nom, décisif sur un corpus français : seule la
  famille E5 est multilingue ici.
- **Mode Serveur HTTP** : délègue les embeddings à un endpoint compatible OpenAI
  (URL + modèle + dimensions). Le bouton **« Tester la connexion »** renvoie la
  dimension réelle du vecteur, à recopier dans le champ Dimensions.
- Case **GPU** : tente CUDA (le runtime GPU est téléchargé automatiquement), avec
  repli CPU silencieux si aucun GPU NVIDIA n'est détecté.

### Reasoning / Chat et Vision

Pour chacun : **URL du serveur**, **modèle**, **clé API** (vide en local),
**activé**, et **effort de raisonnement** (`auto` par défaut = on laisse le
serveur décider). Bouton **« Tester la connexion »**.

### Transcription (audio / vidéo) et Description vidéo

Deux serveurs distincts, désactivés par défaut. **Ollama ne fait pas de
transcription** : les défauts pointent volontairement ailleurs (port 8000, celui
de speaches ; whisper.cpp écoute sur 8080).

Tout ce qui varie d'un serveur à l'autre est réglable : chemin de l'endpoint,
langue, format de réponse, champs supplémentaires, plafond de taille (0 = illimité,
l'envoi est streamé), délai d'attente. Pour la vidéo, le mode de **livraison** :
`base64` (universel) ou `file_uri` (le serveur lit le fichier lui-même — bien plus
efficace en local).

### Ordonnancement de l'indexation

- **Séquentiel** (défaut) : un fichier de bout en bout, puis le suivant. L'index
  avance en continu.
- **Batch** : une tranche de N fichiers passe par tous les étages LLM, puis par
  l'embedding. Un seul échange de modèles par tranche au lieu d'un par fichier —
  décisif si tes modèles ne tiennent pas ensemble en mémoire.

### Classification des dossiers

Un curseur **récursif ↔ bloc** (0 à 1, défaut 0,5) règle l'agressivité du
classement. À gauche : on explore au maximum. À droite : on regroupe plus
volontiers les dossiers techniques. Le changer fait **oublier les classements
existants** — enregistre, puis ré-indexe.

### Qualification du sens (IA)

Quatre interrupteurs indépendants (documents, images, médias, contexte) pour
couper les appels de qualification correspondants, et un réglage d'**effort de
raisonnement** dédié — à `none` par défaut, car ce sont des milliers d'appels
dont la réponse tient en quelques mots.

### Recherche (RAG)

- **Hybride** : fusionne le sens (vecteurs) et les mots-clés (BM25).
- **Reranking** : réordonne le haut du panier avec un cross-encoder local
  (`bge-reranker-v2-m3` par défaut, multilingue).

### Serveurs MCP

Ajoute des outils externes à l'agent (URL HTTP + en-tête d'auth, ou commande
stdio). ⚠️ Ces outils ne sont **pas** bornés à tes dossiers indexés — n'ajoute que
des serveurs de confiance.

### Mémoire de l'agent

Liste ce que l'assistant a retenu ; tu peux oublier une entrée ou tout effacer.

### Prompts

Huit prompts système éditables (classification, description de dossier,
extraction, légende d'image, OCR, description vidéo, assistant, planificateur).
**Un champ vide = le prompt par défaut intégré.**

### Catalogue de modèles

Pour chaque créneau, un catalogue **live** : benchmarks MTEB (embedding) et
OpenCompass (vision/reasoning), plus la **bibliothèque officielle Ollama**. Tu
peux trier par score, popularité ou date, **choisir la quantification** (avec sa
taille réelle) et filtrer sur ce qui tient dans ta VRAM, puis télécharger — ou
supprimer — en un clic.

> ⚠️ Changer le **modèle d'embedding ou ses dimensions** impose une
> **ré-indexation complète** (les vecteurs ne sont plus comparables). Tout le
> reste — reasoning, vision, médias, prompts, RAG, clés API — prend effet
> immédiatement, **sans ré-indexation**.

---

## 9. Récapitulatif des indicateurs

| Indicateur | Où | Signification |
|---|---|---|
| Barre bleue + spinner | Barre latérale | Indexation en cours |
| Barre verte + ✓ | Barre latérale | Index à jour |
| `X à indexer` (ambre) | Barre latérale | Fichiers restant à traiter |
| `N en échec` (rouge) | Barre latérale | Fichiers non traités — cliquer pour agir |
| Pastille 🟢 / 🟡 / 🔴 | Liste de fichiers | Indexé / en attente / échec |
| Pastille de santé | Dossiers racines | Doublons, encombrement, dossiers vides |
| Voyant IA 🟢 / ⚫ | Barre latérale | Créneau prêt / indisponible ou désactivé (un par serveur) |
| Score `NN %` | Résultats de recherche | Pertinence |
| Ligne `🔍 / 📄 / 📂` | Chat | Outil que l'agent utilise en ce moment |

---

## 10. Dépannage

- **L'indexation est très lente** → vérifie que l'effort de raisonnement des
  qualifications est bien à `none` (Paramètres → Qualification du sens). Un modèle
  « thinking » qui réfléchit avant chaque réponse courte peut multiplier le temps
  par trente.
- **La recherche ne renvoie rien** → l'indexation est peut-être en cours, le
  fichier est dans un dossier-bloc, ou ta portée est restreinte au dossier courant.
- **Le chat est grisé** → aucun serveur de reasoning n'est configuré. Lance ton
  runner (ex. `ollama serve`), renseigne son URL, puis « Tester la connexion ».
- **Le chat répond mais ne cherche jamais** → ton modèle ne sait probablement pas
  appeler d'outils. Choisis-en un annoncé comme compatible « tools ».
- **Les images ne sont pas comprises** → active la **Vision** et pointe-la vers un
  vrai modèle multimodal (ex. `moondream`, `qwen2.5vl`, `minicpm-v`).
- **La vision échoue par intermittence** → reasoning et vision se partagent le
  même GPU et s'échangent en mémoire. Utilise un petit modèle de vision, sépare-les
  sur deux machines, ou passe en mode **batch**.
- **Repartir de zéro** → fermer l'app et supprimer le dossier
  `…/AppData/Roaming/com.virgi.sensetree/` (index régénérable ; tes fichiers ne
  sont pas concernés).

---

## 11. Pour aller plus loin

La documentation complète est dans le [wiki du dépôt](https://github.com/Eligrive/SenseTree/wiki) :
architecture, pipeline d'indexation, fonctionnement détaillé du RAG et de l'agent,
et le **protocole exact** de chaque requête envoyée aux serveurs IA.
