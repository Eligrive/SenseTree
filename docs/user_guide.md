# Guide d'utilisation — SenseTree

SenseTree est un **explorateur de fichiers augmenté**, local-first. Il ne remplace
pas l'explorateur de Windows : il se superpose à ton système de fichiers réel et
y ajoute une **couche de compréhension sémantique** (recherche par le sens, chat
IA sur tes fichiers, rangement assisté). Tout tourne **en local** : tes données ne
quittent jamais ta machine.

---

## 1. Ce que fait l'application

| Capacité | Description |
|---|---|
| **Indexation sémantique** | Chaque fichier reçoit un vecteur de sens (embedding) calculé à partir de son **contenu** (texte, PDF, DOCX, image via vision) ou, à défaut, de son **contexte** (nom + dossier + type). |
| **Recherche sémantique** | Retrouver des fichiers par leur signification, pas par mots-clés exacts : « fichiers liés à mon voyage en Corée ». |
| **Chat IA** | Poser des questions sur un dossier, ou demander un rangement en langage naturel. |
| **Rangement assisté (Dry-Run)** | L'IA propose déplacements / renommages / suppressions. **Rien n'est appliqué sans ta validation explicite.** |
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
                                          Texte / Vision / Contexte
```

1. **Crawler** — au démarrage, parcourt tes dossiers racines et met les fichiers
   nouveaux ou modifiés dans une file d'attente (SQLite).
2. **Watchdog** — surveille en temps réel les créations / modifications /
   suppressions et alimente la même file.
3. **Worker** (tâche de fond) — pour chaque fichier :
   extraction du contenu → empreinte SHA-256 (pour ne pas ré-indexer l'identique)
   → découpage en morceaux → **embedding** → stockage dans **LanceDB**.
4. **Recherche** — ta requête est vectorisée puis comparée aux vecteurs stockés ;
   les fichiers dont l'existence est confirmée sur le disque sont renvoyés, triés
   par pertinence.

Trois voies d'« extraction du sens » selon le fichier :

- **Textuelle** : PDF, DOCX, code, `.txt`, Markdown… → contenu réel.
- **Visuelle** : images, et PDF scannés dont aucun texte n'est extractible —
  leurs pages sont alors dessinées puis soumises à un modèle de vision, qui
  en donne une description et en transcrit le texte (si activé).
- **Contextuelle** : fichiers illisibles (machines virtuelles, binaires, gros
  fichiers) → indexés par leur nom, dossier et type. Ils restent donc trouvables.

### Où sont stockées les données

Tout est local, dans `…/AppData/Roaming/com.virgi.sensetree/` :

- `sensetree.sqlite` — catalogue, file d'indexation, journal des actions.
- `lancedb/` — base vectorielle (les embeddings).
- `settings.json` — ta configuration (modèles, serveurs, dossiers).
- `trash/` — corbeille interne (les « suppressions » y sont déplacées, donc réversibles).

---

## 3. Premier lancement

Au tout premier démarrage :

1. Le **modèle d'embedding local** (~130 Mo) se télécharge une fois depuis
   Hugging Face, puis est mis en cache.
2. L'**indexation démarre** en tâche de fond sur ton dossier Documents (dossier
   racine par défaut, modifiable dans les Paramètres).
3. Tu peux utiliser l'app immédiatement ; la recherche devient de plus en plus
   complète à mesure que les fichiers sont indexés.

> ⏳ L'indexation initiale peut être longue (plusieurs milliers de fichiers en
> CPU). Suis l'avancement grâce à l'**indicateur d'indexation** dans la barre de
> gauche.

---

## 4. L'interface

L'écran est divisé en **trois panneaux**.

### 4.1. Barre latérale (gauche)

- **Dossiers indexés** — tes racines. Clique pour naviguer dedans. Le dossier
  actif est surligné en bleu.
- **Indicateur d'indexation** — barre de progression + compteur :
  - `X à indexer` (en ambre) et `Y docs` (total en file).
  - Barre **bleue** avec spinner = indexation en cours ; **verte** avec ✓ = à jour.
  - Une mention rouge « N en échec » apparaît si des fichiers n'ont pas pu être traités.
- **Diagnostic du dossier** — lance l'analyse « gardener » du dossier courant.
- **État IA** — trois pastilles :
  - 🟢 vert = provider connecté / prêt · ⚫ gris = indisponible ou désactivé.
  - `Embedding` (indexation), `Reasoning` (chat), `Vision` (analyse d'images).
  - Survole une pastille pour voir le détail (ex. « connecté », « désactivé »,
    message d'erreur).
- **Paramètres** — configuration des modèles et serveurs.

### 4.2. Explorateur (centre)

- **Barre de recherche sémantique** (en haut) — tape une requête en langage
  naturel et valide. Sous la barre, une case **« Limiter au dossier courant »** :
  - décochée (défaut) = recherche **globale** sur tout l'index ;
  - cochée = recherche restreinte au dossier ouvert.
- **Fil d'ariane** — chemin cliquable pour remonter dans l'arborescence.
- **Liste de fichiers** — colonnes Nom / Taille / Modifié.
  - Icône colorée selon le type (dossier, image, document, texte/code…).
  - **Pastille de statut d'indexation** à droite du nom :
    - 🟢 vert = indexé · 🟡 ambre = en attente · 🔴 rouge = échec · (aucune) = non concerné.
  - **Double-clic** : ouvre le dossier, ou ouvre le fichier avec l'application par défaut.
- **Vue résultats de recherche** — remplace la liste quand une recherche est
  active. Chaque résultat affiche le nom, le chemin, un **score de pertinence
  (%)** et un extrait. Double-clique pour ouvrir. Le lien « retour à
  l'explorateur » revient à la navigation classique.

### 4.3. Assistant / Chat (droite)

- Zone de conversation + champ de saisie (Entrée pour envoyer, Maj+Entrée pour
  aller à la ligne).
- **Deux comportements automatiques** selon ta phrase :
  - **Question** (« résume les documents de ce dossier », « y a-t-il un contrat
    ici ? ») → réponse conversationnelle, avec le contexte du dossier courant.
  - **Instruction d'action** (« range ce dossier », « renomme ces factures »,
    « trie par année ») → génère un **plan Dry-Run** (voir §6).
- Si aucun modèle de reasoning n'est détecté, le champ est désactivé avec un
  message t'invitant à en configurer un.

---

## 5. La recherche sémantique

- Formule ta requête **par le sens** : SenseTree comprend le thème, pas seulement
  les mots exacts. « photos de montagne » peut remonter un fichier `IMG_4821.jpg`
  décrit par la vision comme « paysage alpin enneigé ».
- **Portée** : globale par défaut ; coche « Limiter au dossier courant » pour
  cibler une branche précise du disque.
- **Score** : le pourcentage indique la proximité sémantique. Les résultats sont
  triés du plus pertinent au moins pertinent, un seul résultat par fichier.
- Un résultat n'apparaît que si le fichier **existe toujours** physiquement.

> Si une recherche ne renvoie rien : le fichier n'est peut-être **pas encore
> indexé** (regarde l'indicateur d'indexation), ou il est hors de la portée
> sélectionnée.

---

## 6. Chat & rangement assisté (Dry-Run)

Quand tu demandes une réorganisation, l'assistant renvoie une **carte de plan
d'action** au lieu d'agir directement :

- En-tête « **Plan d'action · Dry-Run** » + nombre d'opérations.
- Chaque ligne montre l'opération :
  - `ancien nom → nouveau nom` (déplacement / renommage) ;
  - `nom (barré) → corbeille` (suppression) ;
  - `nom` avec icône dossier (création de dossier).
- Deux boutons :
  - **Appliquer** — exécute réellement les opérations sur le disque, de façon
    **transactionnelle** : si une opération échoue (fichier verrouillé…),
    **tout est annulé** (rollback) pour ne jamais laisser l'arborescence à moitié
    modifiée. L'index vectoriel est mis à jour sans re-calcul (les fichiers
    déplacés gardent leur sens).
  - **Annuler** — abandonne le plan, rien n'est touché.
- Les **suppressions** ne sont pas destructives : les fichiers sont déplacés dans
  la corbeille interne (`trash/`), donc récupérables.
- Sécurité : un plan qui sortirait de tes dossiers racines configurés est refusé.

---

## 7. Diagnostic « gardener »

Bouton **« Diagnostic du dossier »** (barre latérale) → analyse le dossier courant
et affiche :

- Nombre de fichiers, profondeur maximale, nombre de groupes de doublons.
- **Suggestions** : dossier fourre-tout, arborescence trop profonde, dossiers
  vides, doublons exacts (même contenu, détecté par empreinte SHA-256).
- La liste des doublons et des dossiers vides.

Le diagnostic est **en lecture seule** : il ne fait que constater. Pour agir,
passe par le chat (Dry-Run).

---

## 8. Paramètres (model-agnostic)

SenseTree se connecte au(x) moteur(s) IA de **ton choix**. Tous les endpoints de
chat suivent le standard **compatible OpenAI** (`/v1/...`), donc tu peux brancher
Ollama, LM Studio, un serveur maison sur ton réseau, ou une API externe.

### Embedding (indexation)

- **Mode Local** (défaut) : modèle ONNX embarqué (`multilingual-e5-small` par
  défaut, multilingue). Choisis le modèle dans la liste ; les dimensions se
  règlent automatiquement.
- **Mode Serveur HTTP** : délègue les embeddings à un endpoint compatible OpenAI
  (URL + modèle + dimensions).
- Case **GPU** : réservée à un binaire compilé avec le support CUDA (par défaut
  l'indexation tourne sur CPU, portable partout).

### Reasoning / Chat et Vision

Pour chacun :

- **URL du serveur (base)** — ex. `http://localhost:11434/v1` (Ollama),
  `http://192.168.1.20:1234/v1` (serveur maison), ou l'URL d'une API externe.
- **Modèle** — ex. `llama3.1:8b`, `moondream` (vision), etc.
- **Clé API** — laissée vide pour un serveur local.
- **Activé** — coche pour activer la fonctionnalité.
- Bouton **« Tester la connexion »** — vérifie que le serveur répond avant de sauvegarder.

> ⚠️ Changer le **modèle d'embedding ou ses dimensions** impose une
> **ré-indexation complète** (les vecteurs ne sont plus comparables). Ajouter un
> dossier racine relance un scan ; la surveillance temps réel de la nouvelle
> racine s'active au prochain démarrage.

---

## 9. Récapitulatif des indicateurs

| Indicateur | Où | Signification |
|---|---|---|
| Barre bleue + spinner | Barre latérale | Indexation en cours |
| Barre verte + ✓ | Barre latérale | Index à jour |
| `X à indexer` (ambre) | Barre latérale | Fichiers restant à traiter |
| Pastille 🟢 / 🟡 / 🔴 | Liste de fichiers | Indexé / en attente / échec |
| Pastille IA 🟢 / ⚫ | Barre latérale | Provider prêt / indisponible ou désactivé |
| Score `NN %` | Résultats de recherche | Pertinence sémantique |
| Bandeau ambre (chat) | Panneau de droite | Aucun modèle de reasoning détecté |

---

## 10. Dépannage

- **La recherche ne renvoie rien** → l'indexation est peut-être en cours (voir
  l'indicateur), ou ta portée est restreinte au dossier courant : décoche la case
  pour chercher globalement.
- **Le chat est grisé** → aucun serveur de reasoning n'est détecté. Lance ton
  runner (ex. `ollama serve`) et renseigne son URL dans les Paramètres, puis
  « Tester la connexion ».
- **Les images ne sont pas comprises** → active la **Vision** dans les Paramètres
  et pointe-la vers un modèle multimodal (ex. `moondream`, `llava`).
- **Indexation très lente** → normal en CPU sur de gros volumes ; l'app reste
  utilisable pendant ce temps. Un build avec support GPU accélère nettement.
- **Repartir de zéro** → fermer l'app et supprimer le dossier
  `…/AppData/Roaming/com.virgi.sensetree/` (index régénérable ; tes fichiers ne
  sont pas concernés).
