# Puls-Events — POC d’assistant RAG pour les événements culturels

> **Document de soutenance — état des livrables examinés**  
> Projet OpenClassrooms, Projet 9 : assistant de recommandation d’événements culturels à Dijon et dans un rayon de 20 km autour, fondé sur les données OpenAgenda.

## 1. Objectifs du projet

### Contexte et problématique

La mission confiée par Puls-Events consiste à faciliter la découverte d’événements culturels locaux. Une personne peut poser une question en français courant — par exemple sur un style musical, un lieu ou une période — au lieu de parcourir les fiches une à une.

Un moteur de recherche lexical classique répond moins facilement aux formulations variées. Le principe du **RAG** (*Retrieval-Augmented Generation*, ou génération augmentée par récupération) est de rechercher d’abord des documents pertinents dans un corpus, puis de fournir ce contexte à un modèle de langage chargé de rédiger une réponse. L’objectif est de rendre les réponses plus utiles tout en les ancrant dans des fiches d’événements identifiables. Le RAG ne garantit cependant pas, à lui seul, l’exactitude : un résultat manquant ou mal classé peut encore mener à une réponse incomplète ou erronée.

### Objectif du POC

Le prototype met en œuvre les étapes suivantes :

1. récupérer des événements depuis l’API OpenAgenda ;
2. aplatir, nettoyer, filtrer et catégoriser les fiches ;
3. encoder les textes, construire un index FAISS et lui associer des métadonnées ;
4. rechercher les événements proches d’une question et fournir leur contexte à Mistral via LangChain ;
5. exposer des fonctions de question-réponse et de reconstruction via FastAPI ;
6. évaluer le système sur un jeu annoté à l’aide de RAGAS.

Le dépôt examiné comprend le code, les données locales, un index et un benchmark. Ces éléments permettent de décrire les livrables et le fonctionnement prévu, mais ne constituent pas à eux seuls une validation d’un déploiement reproductible.

### Périmètre des données

La zone visée est **Dijon et 20 km autour**. `config/settings.py` définit un rayon configurable par `RADIUS_KM` (20 km par défaut) ; `config/geo.py` fixe le centre géographique de Dijon et applique le calcul de distance lorsque les coordonnées sont disponibles. Le filtre comporte aussi des critères de repli : Dijon, le département de la Côte-d’Or ou un code postal commençant par `21`. Par conséquent, lorsque ces critères textuels sont utilisés, le résultat peut dépasser le cercle strict de 20 km.

La configuration expose `DATE_MIN` et `DATE_MAX` en fonction de la date de référence (un an avant et un an après), avec `REFERENCE_DATE` pour reproduire une date de référence. Toutefois, la version examinée de la collecte et du nettoyage ne se sert pas de `DATE_MAX` comme borne supérieure : les événements futurs au-delà d’un an peuvent être conservés. La borne historique reste définie à un an.

## 2. Architecture du système

### Schéma logique

```text
                         collecte de données
OpenAgenda API ──► recherche d'agendas, pagination et filtre géographique
                                      │
                                      ▼
                        événements bruts (JSON)
                                      │
                                      ▼
             aplatissement, nettoyage, dates UTC, catégorisation
                                      │
                                      ▼
                       événements préparés (Parquet)
                                      │
                   BAAI/bge-m3, dimension 1024, lots de 16
                                      │
                                      ▼
                        index FAISS IndexFlatL2
                         + métadonnées Parquet

Question ──► embedding de la question ──► recherche FAISS (k=5)
                                               │
                                               ▼
                                     contexte des événements
                                               │
                                               ▼
                              prompt LangChain ──► Mistral
                                               │
                                               ▼
                                      réponse en français

Jeu annoté ──► récupération + génération ──► évaluation RAGAS ──► CSV
```

Il s’agit du flux décrit par les scripts. Le dépôt organise les composants en packages et répertoires dédiés (`api/`, `config/`, `scripts/`, `tests/` et `ui/`) ; l’arborescence constatée figure en section 9.

### Outils principaux

- **Python** (version minimale déclarée : 3.10) et `pyproject.toml` / setuptools ;
- **OpenAgenda API** et `requests` pour la collecte ;
- **pandas**, **PyArrow** et Parquet pour le traitement et les fichiers tabulaires ;
- **BeautifulSoup** pour le nettoyage du texte HTML ;
- **Sentence Transformers** et `BAAI/bge-m3` pour les embeddings (1024 dimensions) ;
- **FAISS CPU** pour la recherche vectorielle ;
- **LangChain** et `ChatMistralAI` pour la génération ;
- **FastAPI**, **Uvicorn** et **APScheduler** pour l’API et la planification ;
- **RAGAS**, un modèle Ollama et des embeddings Hugging Face dans le script d’évaluation ;
- **Streamlit** et `requests` pour l’interface définie dans `ui/app.py` ;
- **Docker** pour l’image décrite dans `Dockerfile` ; **pytest** pour les tests.

## 3. Préparation et vectorisation des données

### Collecte OpenAgenda

`scripts/fetch_events.py` interroge `https://api.openagenda.com/v2`. Le script recherche des agendas liés à Dijon, parcourt les résultats paginés, puis retient les agendas dont le titre correspond à la liste blanche configurée dans `config/settings.py` (notamment Dijon, Côte-d’Or, Bourgogne, La Vapeur et atheneum). Les événements sont filtrés selon leurs coordonnées, leur ville, leur département ou leur code postal, puis dédupliqués par identifiant `uid` avant l’écriture du JSON brut.

La fenêtre de collecte utilise la date de référence définie par `REFERENCE_DATE` si elle est valide, ou la date courante sinon. `DATE_MIN` est la borne historique ; bien que `DATE_MAX` soit déclarée dans les paramètres, la collecte ne l’utilise pas comme borne haute. La configuration prévoit une taille de page de 100, jusqu’à quatre tentatives et une pause de 0,2 seconde après une réponse réussie. Une clé OpenAgenda est requise à l’exécution. **Aucune collecte en direct ni aucun appel de service externe n’a été exécuté pour préparer ce README.**

### Nettoyage et structuration

`scripts/clean_events.py` aplatit les champs multilingues en privilégiant le français, extrait les dates depuis `timings` puis depuis des formats de repli, nettoie le HTML et convertit les dates en UTC. Le traitement déduplique par `uid`, exige un titre et une description ou une description longue, filtre les événements qui ne chevauchent pas la période conservée et applique le filtre géographique. Il attribue des catégories à partir de mots-clés. Les catégories `sante_bien_etre` et `vivre_ensemble` sont déclarées hors périmètre et exclues lorsqu’elles sont la catégorie principale.

Le script construit un champ textuel de recherche comportant, selon les données disponibles, le titre, les catégories, les descriptions, les dates, le lieu, les mots-clés et les conditions. Les métadonnées contiennent notamment l’identifiant, les dates, le lieu, l’URL et les catégories.

### Corpus Parquet et index effectivement fournis

Le fichier `data/processed/events.parquet` contient **1123 lignes et 24 colonnes**, dont `uid`, `title`, `description`, `long_description`, `keywords`, `date_begin`, `date_end`, des champs de lieu, `url`, `categories`, `category`, `categories_txt` et `text`. Les 1 080 identifiants sont distincts et non nuls ; aucune valeur nulle n’a été observée dans les colonnes de ce fichier. La longueur du texte de recherche va de **136 à 6 112 caractères**, avec une moyenne d’environ **989912 caractères**. Les dates `date_begin` vont du **5 novembre 2025** au **1er janvier 2032**.

Le fichier livré `data/vector_store/metadata.parquet` contient également 1 080 lignes et 24 colonnes. Dans les artefacts examinés, ses valeurs correspondent intégralement, colonne par colonne et dans le même ordre, à celles de `events.parquet`. L’index fourni contient **1 080 vecteurs** : son nombre de vecteurs concorde avec le nombre de lignes. Ces chiffres décrivent les artefacts présents ; ils ne prouvent pas à eux seuls leur fraîcheur ni la date précise de leur génération.

Au **5 novembre 2026**, 973 des 1 080 dates `date_begin` sont antérieures à cette date et 107 sont égales ou postérieures. La plage de dates, qui s’étend jusqu’en 2032, reflète la nature des données fournies et la logique actuelle sans borne supérieure ; elle ne constitue pas la fenêtre d’un an que pourrait laisser supposer `DATE_MAX`. Une collecte fraîche n’a pas été réalisée au cours de cette vérification.

### Découpage (chunking)

Le pipeline n’applique pas de découpage textuel supplémentaire. Chaque événement constitue une unité documentaire : les champs sont consolidés dans un texte unique, transformé en embedding puis indexé dans FAISS. Lors d’une recherche, les événements les plus pertinents sont récupérés et leurs textes assemblés pour constituer le contexte du modèle génératif.

### Embeddings : modèle, dimensions et taille des lots

Le script de vectorisation encode le texte construit par le nettoyage avec `BAAI/bge-m3`, en lots de **16**, convertit les vecteurs en `float32` et les ajoute à un `faiss.IndexFlatL2`. BGE-M3 produit des embeddings denses de **1024 dimensions**. La configuration limite la longueur des séquences à 512 tokens pour maîtriser l’usage mémoire (le modèle peut accepter des séquences plus longues) et normalise les vecteurs avant indexation et recherche. Aucun préfixe d’instruction propre à BGE-Small n’est ajouté aux requêtes. Après ce changement de modèle, relancez `make vectorize` (après avoir le Parquet nettoyé) pour reconstruire index et métadonnées avant de lancer l’API.

L’index FAISS fourni a **384 dimensions** et est donc incompatible avec les vecteurs BGE-M3 de **1024 dimensions**. Il faut reconstruire l’index et ses métadonnées avec le nouveau modèle avant d’utiliser le retriever ; les vecteurs de l’ancien index ne peuvent pas être convertis en changeant simplement sa dimension. Le code signale désormais explicitement un désaccord de dimensions.

Le nettoyage crée un champ `text` et le vectoriseur le réutilise directement : le texte des métadonnées enregistrées est donc celui que le script encode. Le retriever vérifie la présence de ce champ et l’égalité entre le nombre de lignes de métadonnées et de vecteurs. Cette vérification ne prouve pas, à elle seule, l’alignement sémantique de chaque ligne et de chaque vecteur, ni la provenance de l’index.

## 4. Choix du modèle NLP

### Modèles sélectionnés et rôle

Deux composants de modèle sont distincts :

- **Recherche sémantique** : `BAAI/bge-m3` (1024 dimensions), chargé localement par Sentence Transformers pour encoder les fiches et les questions ;
- **Génération** : `open-mistral-7b`, configuré par `ChatMistralAI` dans `api/mistral_qa.py`, avec une température de **0,2**. L’appel de génération utilise l’API Mistral et nécessite une clé fournie à l’exécution.

Le prompt indique la date du jour, demande une réponse en français, fournit le contexte récupéré, demande de signaler si la réponse n’y figure pas et interdit de mentionner un événement absent du contexte. Il indique aussi de ne pas supposer de période temporelle non précisée par l’utilisateur et de lister les événements à différentes dates sans filtrage additionnel.

### Évaluation et limites des modèles

Dans `evaluate_rag.py`, le modèle d’évaluation RAGAS est **Ollama `mistral`** par défaut ; les embeddings d’évaluation sont également locaux, via `HuggingFaceEmbeddings` et `BAAI/bge-m3`. Ce modèle d’évaluation ne remplace pas le modèle Mistral utilisé pour générer les réponses du système RAG. La configuration du script et la disponibilité des services locaux influent sur la reproduction d’une évaluation.

Les limites à garder en vue sont la dépendance de la génération à l’API Mistral, à sa disponibilité et à ses conditions d’utilisation ; la dépendance du résultat à la qualité, à la couverture et à la fraîcheur des fiches ; et le risque qu’une récupération incomplète produise une réponse erronée ou un faux constat d’absence. Aucun benchmark de latence réelle, de coût ou de comparaison entre modèles n’est documenté par les artefacts examinés.

## 5. Construction de la base vectorielle

### FAISS et persistance

Le code utilise un index FAISS exact `IndexFlatL2`, enregistré par `faiss.write_index`, ainsi qu’un fichier Parquet pour les métadonnées. Les chemins déclarés par `config/settings.py` sont :

```text
data/raw/events_raw.json
data/processed/events.parquet
data/vector_store/faiss_index.bin
data/vector_store/metadata.parquet
```

Les fichiers livrés se trouvent à ces emplacements dans le dépôt. Les données locales et les artefacts générés sont ignorés par `.gitignore` selon les règles déclarées ; ils sont destinés à être régénérés plutôt qu’à être systématiquement versionnés.

Le retriever vérifie que le champ `text` existe dans les métadonnées et que `index.ntotal == len(metadata)`. Ces conditions sont satisfaites dans les artefacts fournis : **1 080 = 1 080**. Les métadonnées contiennent notamment `uid`, `title`, `location_name`, `date_begin` et le texte ; les documents retrouvés portent aussi une distance L2. Le code utilise `BAAI/bge-m3` pour indexer et encoder les requêtes, en 1024 dimensions ; l’index livré, en 384 dimensions, doit être reconstruit. Le retriever contrôle désormais la dimension de chaque embedding de requête contre celle de l’index, et signale une incompatibilité ; l’égalité des cardinalités ne suffit toutefois pas à démontrer la correspondance ligne-vecteur.

Le retriever calcule `k_effective = min(k, ntotal)` ; la génération demande `k=5` par défaut. Aucun seuil de distance n’est appliqué. Le score L2 ne doit pas être interprété comme une probabilité de pertinence.

## 6. API et interface

### Framework et routes implémentées dans `api/main.py`

Le fichier `api/main.py` définit une application FastAPI avec les routes suivantes :

| Route | Rôle déclaré dans le code |
|---|---|
| `GET /health` | Retourne un état, la dernière reconstruction connue et la prochaine reconstruction planifiée. |
| `POST /ask` | Accepte un champ `query` non vide et retourne `{"response": "…"}`. |
| `POST /rebuild` | Programme le pipeline de collecte, de nettoyage et de vectorisation en tâche de fond. |
| `POST /rebuild/now` | Lance le pipeline immédiatement et attend son résultat. |
| `GET /rebuild/status` | Retourne l’état du verrou, les dates de reconstruction et l’état de l’ordonnanceur. |

Le modèle Pydantic de requête est `QueryRequest`, avec `query: str` et une longueur minimale de 1. La route `/ask` rejette également une chaîne composée d’espaces. Les erreurs inattendues de génération sont journalisées et transformées en HTTP 500 ; le détail de l’exception est inclus dans la réponse HTTP par le code actuel. L’ordonnanceur déclare une reconstruction hebdomadaire le lundi à 03 h 00, selon le fuseau horaire par défaut du processus.

La réponse de `/ask` ne renvoie que la chaîne `response` ; elle n’expose pas de liste de sources séparée dans le JSON. L’interface `ui/app.py` est une application Streamlit qui envoie `{"query": …}` à `/ask`, à l’URL par défaut `http://localhost:8000` (surcharge possible par `PULS_API_URL`), et accepte plusieurs noms de champ de réponse.

### Exemple de requête conforme au contrat de l’API

Lorsque l’API est disponible :

```bash
curl -X POST http://localhost:8000/ask \
  -H 'Content-Type: application/json' \
  -d '{"query":"Quels concerts sont prévus à La Vapeur en novembre 2026 ?"}'
```

Exemple de forme attendue par le code (le contenu réel dépend des données et des services) :

```json
{"response": "Réponse rédigée à partir du contexte récupéré…"}
```

### Lancement déclaré et reproductibilité

Le Makefile déclare `make api` (`uvicorn api.main:app --reload`) et `make ui` (`streamlit run ui/app.py`). Les répertoires et fichiers correspondants figurent dans le dépôt. Les commandes et chemins sont donc vérifiables statiquement, mais leur présence ne garantit pas que l’application démarre dans un environnement donné. Aucune commande de lancement n’a été exécutée avec succès ni revendiquée pendant cette vérification.

## 7. Évaluation du système

### Jeu de questions annotées et schéma

Le fichier `annotated_qa.json` contient **42 questions uniques**. Chaque entrée possède les trois clés `question`, `ground_truth` et `reference_contexts` ; les champs de référence sont présents pour les 42 entrées.

Le fichier `ragas_evaluation_results.csv` contient **42 lignes**, avec les colonnes `user_input`, `retrieved_contexts`, `reference_contexts`, `response`, `reference`, `faithfulness`, `answer_relevancy` et `context_precision`. Les 42 questions du CSV correspondent exactement aux questions du jeu annoté. Le script `tests/evaluate_rag.py` configure une limite d’évaluation de **42** cas, avec `RETRIEVAL_K = 5`.

Le Makefile lance `python evaluate_rag.py`. L’évaluation nécessite la disponibilité des dépendances RAGAS, du service Ollama configuré et de l’index reconstruit en 1024 dimensions ; la commande n’a pas été exécutée ici. Le CSV ne documente pas à lui seul l’environnement exact ni les versions ayant produit les scores.

Le schéma attendu comprend une question, `ground_truth` et `reference_contexts`, des contextes récupérés et une réponse générée. Pour RAGAS récent, `ground_truth` est transmis sous le nom `reference`, employé par `ContextRecall` ; pour l’ancien schéma, il est transmis sous `ground_truth`. La présence d’un champ de référence ne suffit pas à décrire entièrement le protocole d’annotation ; l’échantillon demeure limité et ne représente pas toutes les requêtes possibles.

### Métriques et résultats du CSV fourni

Les moyennes ci-dessous proviennent du CSV existant sont à recalculer après une nouvelle évaluation (aucune valeur manquante n’est remplacée par zéro) :

| Métrique | Valeurs disponibles | Moyenne (n valide) |
|---|---:|---:|
| `faithfulness` | 39 valeurs sur 42 | **0,8346 (n = 39)** |
| `answer_relevancy` | 42 valeurs sur 42 | **0,6358 (n = 42)** |
| `context_precision` | 41 valeurs sur 42 | **0,8317 (n = 41)** |

Le CSV ne présente pas de lignes permettant de reprendre les exemples décrits dans la version précédente du README comme un benchmark de trois questions (notamment le cas « jazz à La Vapeur »). Les 42 cas sont ceux du fichier `annotated_qa.json`. À titre d’exemple, la question sur le volume de la série *Roads* dans la fiche de Thylacine a `faithfulness = 0,5`, `answer_relevancy ≈ 0,6567` et `context_precision ≈ 0,7000`. Le cas sur l’œuvre EX(S)ISTERE affiche `faithfulness = 0`, `answer_relevancy = 0` et `context_precision ≈ 0,8056` ; la réponse évaluée indique que l’information ne figure pas dans le contexte, alors que la référence est renseignée. Ces observations illustrent des limites de récupération ou de réponse, mais ne permettent pas à elles seules d’en établir la cause.

 Les métriques RAGAS ne constituent pas une mesure directe de satisfaction utilisateur, et 42 observations ne permettent pas de généraliser à la qualité globale du système.

## 8. Recommandations et perspectives

### Points effectivement observables

- Une chaîne de traitement de bout en bout est représentée dans les scripts : collecte, nettoyage, vectorisation, recherche et génération.
- Le dépôt comprend un corpus de **1 080 fiches**, des métadonnées correspondantes et un index FAISS de **1 080 vecteurs**.
- Le jeu annoté comporte 42 cas avec réponses et contextes de référence ; le prompt demande explicitement de ne pas inventer d’événement absent du contexte.
- L’API et l’interface définissent des contrats simples, ainsi que des routes de statut et de reconstruction.

Ces constats décrivent les fichiers examinés ; ils ne remplacent pas l’exécution reproductible de l’application et des tests.

### Limites établies par les artefacts

- **Fraîcheur et périmètre temporel :** la date exacte de production des Parquet n’est pas établie. Au 5 novembre 2026, 973 des 1 080 dates de début sont antérieures à cette date et 107 sont égales ou postérieures. La plage s’étend jusqu’en 2032, car la configuration `DATE_MAX` n’est pas appliquée comme borne supérieure dans la collecte et le nettoyage examinés. Une collecte fraîche n’a pas été effectuée lors de cette vérification.
- **Périmètre géographique :** le rayon par défaut est bien de 20 km autour de Dijon, mais les critères de repli par ville, département et code postal peuvent inclure des événements situés hors du cercle.
- **Embeddings et index :** le modèle configuré est `BAAI/bge-m3` (1024 dimensions), tandis que l’index fourni est en 384 dimensions ; une reconstruction complète de l’index est requise avant le retrieval. Les embeddings sont normalisés, sans préfixe de requête BGE-Small.
- **Évaluation :** les scores disponibles concernent 42 questions. Le cas EX(S)ISTERE affiche un score de fidélité nul et un score de pertinence de réponse nul ; la valeur de `context_recall` reste à recalculer ; aucune mesure de satisfaction, et les scores disponibles ne mesurent pas la fraîcheur, la latence ou le coût.
- **Docker :** le Dockerfile utilise `python:3.12-slim`, copie `api/`, `config/`, `scripts/` et `ui/`, et lance Uvicorn sur le port 8000. Le `.dockerignore` exclut notamment `.env`, les tests et le CSV RAGAS du contexte d’image ; il n’exclut pas explicitement `data/`, mais le Dockerfile ne copie pas ce répertoire dans l’image. Le modèle `BAAI/bge-m3` (environ 2,2 Go de fichiers de modèle) est pré-téléchargé pendant la construction Docker ; aucun téléchargement ni build n’a été exécuté dans cette vérification. Aucun `docker build` ni lancement de conteneur n’a été exécuté : le fonctionnement de l’image et la disponibilité des données au runtime ne sont donc pas établis.
- **Dépendances :** `pyproject.toml` déclare Python `>=3.10`. La dépendance `truststore` est présente dans le groupe `core`, contrairement à ce que disait l’ancienne version du README. Les dépendances d’évaluation et de l’interface sont regroupées dans les extras `dev` et `ui`.
- **Réseau et secrets :** la génération dépend d’une clé Mistral ; la collecte dépend d’une clé OpenAgenda. Les clés doivent rester dans l’environnement local et ne jamais être publiées. La collecte utilise `verify=False` dans la requête HTTP examinée, ce qui constitue un point de vigilance de sécurité.

### Perspectives

Pour poursuivre le projet, il serait utile de stabiliser et documenter les versions des composants, de préciser la provenance et la date des corpus, et d’élargir l’évaluation à des cas positifs, négatifs, ambigus et hors périmètre. Il serait également pertinent d’observer séparément la couverture des résultats, la fidélité factuelle, la latence et le coût, puis de rendre les reconstructions vérifiables et sûres en déploiement.

Ces éléments sont présentés comme **limites et perspectives** pour cadrer le POC ; ce README ne prétend pas qu’ils ont été réalisés ni qu’un test ou une commande de déploiement a réussi.

## 9. Organisation du dépôt et commandes

### Arborescence constatée

L’arborescence ci-dessous reprend les principaux éléments constatés dans le dépôt au 5 novembre 2026. Les environnements virtuels, caches et répertoires internes de Git sont omis.

```text
DS_OC_Projet9/
├── api/
│   ├── __init__.py
│   ├── faiss_retriever.py
│   ├── main.py
│   └── mistral_qa.py
├── config/
│   ├── __init__.py
│   ├── geo.py
│   └── settings.py
├── data/
│   ├── last_rebuild.json
│   ├── processed/
│   │   └── events.parquet
│   ├── raw/
│   │   └── events_raw.json
│   └── vector_store/
│       ├── faiss_index.bin
│       └── metadata.parquet
├── docs/
│   ├── Cours RAG pour LLM.pdf
│   ├── pwp_soutenance_P9_RAG.pptx
│   └── Template+de+rapport+technique.docx
├── scripts/
│   ├── __init__.py
│   ├── clean_events.py
│   ├── fetch_events.py
│   ├── files_opener.ipynb
│   └── vectorize_events.py
├── tests/
│   ├── __init__.py
│   ├── evaluate_rag.py
│   ├── test_api.py
│   ├── test_clean_events_units.py
│   ├── test_fetch_events.py
│   ├── test_pipeline.py
│   ├── test_rebuid_api.py
│   └── test_vectorize_events.py
├── ui/
│   └── app.py
├── .dockerignore
├── .gitignore
├── annotated_qa.json
├── Dockerfile
├── Makefile
├── pyproject.toml
├── ragas_evaluation_results.csv
└── README.md
```

Le nom `test_rebuid_api.py` est reproduit tel qu’il existe dans le dépôt (coquille dans le nom de fichier). Le fichier brut est `data/raw/events_raw.json` — et non un Parquet —, le corpus nettoyé est `data/processed/events.parquet`, et les artefacts vectoriels sont `data/vector_store/faiss_index.bin` et `data/vector_store/metadata.parquet`.

### Installation déclarée

Le projet déclare Python **3.10 ou supérieur** ; le Dockerfile s’appuie sur Python **3.12-slim**. Depuis la racine du dépôt, les commandes d’installation inscrites au Makefile sont :

```bash
python -m venv .venv
source .venv/bin/activate
make install       # pip install -e ".[core]"
make install-dev   # ajoute les dépendances de développement et d’évaluation
make install-ui    # ajoute Streamlit et requests
```

Sous Windows, le Makefile recommande WSL, Git Bash ou l’exécution des commandes pip équivalentes. Pour lancer la collecte, configurez localement les variables requises, notamment `OPENAGENDA_API_KEY` et, pour la génération, `MISTRAL_API_KEY`. Les valeurs sont à fournir à l’environnement d’exécution ; **aucune clé ne doit être inscrite dans un fichier partagé**. Exemple de fichier d’environnement local, avec valeurs fictives uniquement :

```dotenv
OPENAGENDA_API_KEY=...
MISTRAL_API_KEY=...
```

### Commandes du Makefile et état de vérification

Les commandes ci-dessous sont celles déclarées dans le Makefile ; elles ne sont pas présentées comme ayant été exécutées avec succès :

| Cible | Commande inscrite | Observation |
|---|---|---|
| `make fetch` | `python -m scripts.fetch_events` | Lance la collecte ; requiert la clé OpenAgenda et un accès réseau. |
| `make clean-data` | `python -m scripts.clean_events` | Nettoie et structure le JSON brut. |
| `make vectorize` | `python -m scripts.vectorize_events` | Construit l’index FAISS et les métadonnées à partir du Parquet nettoyé. |
| `make pipeline` | `fetch`, puis `clean-data`, puis `vectorize` | Enchaîne les trois étapes précédentes ; la collecte est une opération réseau. |
| `make api` | `uvicorn api.main:app --reload` | Lance le serveur de développement FastAPI. |
| `make ui` | `streamlit run ui/app.py` | Lance l’interface Streamlit. |
| `make test` | `pytest tests/ -v --tb=short --cov=scripts --cov=api --cov-report=term-missing` | Lance les tests présents dans `tests/` et produit un rapport de couverture. Aucun résultat de test n’est revendiqué ici. |
| `make evaluate` | `python evaluate_rag.py` | Lance le script d’évaluation présent à la racine. |
| `make docker-build` | `docker build -t puls-events-rag .` | Construit l’image ; aucune construction n’a été exécutée. |
| `make docker-run` | `docker run --rm -p 8000:8000 --env-file .env --mount "type=bind,source=$(CURDIR)/data,target=/app/data" puls-events-rag` | Requiert une image construite, un fichier `.env` local et les données runtime. |

La cible `make all` enchaîne le pipeline, les tests, l’évaluation et la construction Docker ; elle n’est pas un contrôle hors ligne et n’a pas été exécutée. `make clean` supprime des caches, fichiers de couverture et CSV à la racine ; `make purge-data` supprime le répertoire `data/` et impose une nouvelle collecte. Utilisez cette dernière cible avec prudence.

## 10. Annexes

### Exemple de structure d’une entrée annotée

Le schéma constaté dans `annotated_qa.json` est :

```json
{
  "question": "À quelle heure commence le concert de Ghinzu du 16 octobre à La Vapeur ?",
  "ground_truth": "La fiche indique un début à 20 h 30, le vendredi 16 octobre.",
  "reference_contexts": ["Extrait de la fiche d’événement…"]
}
```

Le fichier contient 42 entrées avec ces trois champs. Le CSV d’évaluation présente les mêmes questions.

### Exemple de prompt et de réponse API

Le prompt réel est défini dans `api/mistral_qa.py` et comprend la date, une consigne de réponse en français, le contexte récupéré et la question. Son principe est de répondre à partir des seuls extraits et de signaler quand l’information n’y figure pas. La réponse HTTP renvoyée par `/ask` est de la forme `{"response": "…"}` ; le code n’inclut pas de tableau de citations indépendant dans cette réponse.

### Vérifications effectuées pour ce README

- Lecture intégrale du README source et du relevé `clipboard.txt` de l’arborescence du dépôt.
- Lecture statique du `Makefile`, du `Dockerfile`, de `.dockerignore`, de `.gitignore`, de `pyproject.toml` et des modules de collecte, de nettoyage, de géographie, de vectorisation, de retrieval, de génération, de l’API, de l’interface et de l’évaluation.
- Vérification des chemins de données, des paramètres géographiques et temporels, des routes API, des commandes du Makefile, de la version Python déclarée et des dépendances.
- Analyse du jeu `annotated_qa.json` et du CSV d’évaluation ; vérification de l’égalité des ensembles de questions et recalcul des moyennes en excluant les valeurs manquantes.
- Lecture des schémas et contenus de `events.parquet` et `metadata.parquet` ; vérification de l’égalité complète des lignes et des colonnes.
- Le CSV RAGAS fourni a 42 lignes ; son en-tête a été inspecté. La lecture binaire FAISS n’a pas pu être refaite dans cet environnement, donc la dimension 384 reprise pour l’index fourni provient des documents README/audit antérieurs.

Aucun appel API, aucune collecte, aucun appel Mistral/Ollama, aucune nouvelle évaluation RAGAS, aucun test pytest et aucune construction ou exécution Docker n’ont été réalisés pour rédiger ce document.
