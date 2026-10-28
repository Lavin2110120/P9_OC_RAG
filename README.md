# Puls-Events — POC d’assistant RAG pour les événements culturels

> **Document de soutenance — état des livrables examinés**  
> Projet de recommandation d’événements culturels autour de Dijon et en Côte-d’Or, fondé sur les données OpenAgenda. Ce dépôt décrit un **prototype** ; les résultats ci-dessous sont ceux des fichiers fournis, et non une garantie de fonctionnement reproductible de toute la chaîne depuis une installation neuve.

## 1. Objectifs du projet

### Contexte et problématique

La mission confiée par Puls-Events est de faciliter la découverte d’événements culturels locaux en permettant à une personne de poser une question en français courant — par exemple sur un style musical, un lieu ou une période — plutôt que de parcourir des fiches une à une.

Un moteur de recherche lexical classique répond moins facilement aux formulations variées. Le principe du **RAG** (*Retrieval-Augmented Generation*) est de rechercher d’abord des documents pertinents dans un corpus, puis de fournir ce contexte à un modèle de langage chargé de rédiger une réponse. L’objectif est de rendre les réponses plus utiles tout en les ancrant dans des fiches d’événements identifiables. Le RAG ne garantit cependant pas, à lui seul, l’exactitude : un résultat manquant ou mal classé peut encore mener à une réponse incomplète ou erronée.

### Objectif du POC

Le prototype vise à mettre en œuvre les étapes suivantes :

1. récupérer des événements depuis l’API OpenAgenda ;
2. aplatir, nettoyer, filtrer et catégoriser les fiches ;
3. encoder les textes, construire un index FAISS et lui associer des métadonnées ;
4. rechercher les événements proches d’une question et fournir leur contexte à Mistral via LangChain ;
5. exposer des fonctions de question-réponse et de reconstruction via FastAPI ;
6. évaluer quelques réponses à l’aide d’un jeu annoté et de RAGAS.

Les fichiers transmis permettent d’examiner les composants, des données sérialisées, un index et un petit benchmark. Ils ne constituent pas, à eux seuls, une validation d’un déploiement complet reproductible. Comme les pièces jointes ont été aplaties lors de leur transmission, leur présentation ne permet pas de conclure que les répertoires attendus sont absents du dépôt source.

### Périmètre des données

La configuration vise Dijon et la Côte-d’Or, avec un rayon géographique par défaut de 20 km autour du centre de Dijon. Le filtre accepte également Dijon, le département 21 ou les codes postaux commençant par `21` ; le périmètre n’est donc pas un cercle strict de 20 km. La fenêtre de collecte et de nettoyage est calculée par rapport à la date d’exécution (un an avant et un an après), avec `REFERENCE_DATE` comme possibilité de figer la date de référence.

## 2. Architecture du système

### Schéma logique

```text
                         collecte de données
OpenAgenda API ──► recherche d’agendas, pagination et filtre géographique
                                      │
                                      ▼
                           événements bruts JSON
                                      │
                                      ▼
             aplatissement, nettoyage, dates UTC, catégorisation
                                      │
                                      ▼
                       événements préparés (Parquet)
                                      │
                   BAAI/bge-small-en-v1.5, batch de 32
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

Il s’agit du flux décrit par les scripts. Les imports, le Makefile et le Dockerfile définissent une arborescence cible — notamment `api/`, `config/`, `scripts/`, `tests/` et `ui/` — présentée en section 9. Les fichiers joints ayant été aplatis, cette forme de transmission ne permet pas d’établir si ces répertoires existent ou non dans le dépôt source.

### Technologies identifiées dans les livrables

- **Python** (version minimale déclarée : 3.10) et `pyproject.toml` / setuptools ;
- **OpenAgenda API** et `requests` pour la collecte ;
- **pandas**, **PyArrow** et Parquet pour le traitement et les fichiers tabulaires ;
- **BeautifulSoup** pour le nettoyage du texte HTML ;
- **Sentence Transformers** et `BAAI/bge-small-en-v1.5` pour les embeddings ;
- **FAISS CPU** pour la recherche vectorielle ;
- **LangChain** et `ChatMistralAI` pour la génération ;
- **FastAPI**, **Uvicorn** et **APScheduler** pour l’API et la planification ;
- **RAGAS**, un modèle Ollama et des embeddings Hugging Face dans le script d’évaluation ;
- **Streamlit** et `requests` pour l’interface présente dans `app.py` ;
- **Docker** pour l’image décrite dans `Dockerfile` ; **pytest** pour les tests déclarés.

## 3. Préparation et vectorisation des données

### Collecte OpenAgenda

`fetch_events.py` interroge `https://api.openagenda.com/v2`. Le code recherche les agendas avec les requêtes `Dijon`, `Côte-d'Or` et `Bourgogne`, parcourt les pages d’agendas et d’événements, puis retient les agendas dont le titre correspond à une liste blanche (par exemple Dijon, Bourgogne, La Vapeur et atheneum). Les événements sont filtrés selon leurs coordonnées, leur ville, leur département ou leur code postal, puis dédupliqués par identifiant `uid` avant écriture en JSON.

La fenêtre de dates est définie dans `settings.py` à partir de la date du jour, ou de `REFERENCE_DATE` si elle est valide. Les paramètres configurés comprennent une taille de page de 100, jusqu’à quatre tentatives et une pause de 0,2 seconde après une réponse réussie. La collecte requiert une clé d’accès OpenAgenda dans l’environnement d’exécution. **Aucune collecte live ni aucun appel de service externe n’a été exécuté pour préparer ce README.**

### Nettoyage et structuration

`clean_events.py` aplatit les champs multilingues en privilégiant le français, extrait les dates depuis `timings`, puis depuis des formats de repli, nettoie les balises HTML, et convertit les dates en UTC. Le traitement déduplique par `uid`, exige un titre et une description ou description longue, filtre les événements qui chevauchent la fenêtre de dates et la zone cible, puis attribue des catégories à partir de mots-clés. Les catégories `sante_bien_etre` et `vivre_ensemble` sont déclarées hors périmètre et exclues lorsqu’elles sont la catégorie principale.

Le script construit un champ textuel de recherche comportant, selon disponibilité, le titre, les catégories, les descriptions, les dates, le lieu, les mots-clés et les conditions. Les métadonnées portent notamment l’identifiant, les dates, le lieu, l’URL et les catégories.

### Corpus Parquet effectivement fourni

Le fichier `events.parquet` a été lu directement. Il contient **833 lignes et 24 colonnes**, dont `uid`, `title`, `description`, `long_description`, `keywords`, `date_begin`, `date_end`, des champs de lieu, `url`, `categories`, `category`, `categories_txt` et `text`. Les 833 identifiants sont distincts et non nuls ; aucune valeur nulle n’a été observée dans les colonnes de ce fichier. Le texte de recherche compte de **175 à 6 112 caractères** (moyenne : **930,7**). Les dates `date_begin` valides vont du **31 octobre 2023** au **17 juin 2027**.

Répartition de la catégorie principale du corpus fourni :

| Catégorie | Événements |
|---|---:|
| autre | 757 |
| musique | 54 |
| sport | 8 |
| biodiversite | 6 |
| theatre | 5 |
| festival | 3 |
| **Total** | **833** |

`metadata.parquet` contient également 833 lignes et 24 colonnes ; ses valeurs correspondent, colonne par colonne et dans le même ordre, à celles de `events.parquet`. L’index fourni contient lui aussi 833 vecteurs : le nombre de lignes et celui des vecteurs concordent. Le fichier `chunks.parquet`, distinct du corpus utilisé par le code de vectorisation décrit ci-dessous, contient **94 lignes et 6 colonnes** (`uid`, `event_uid`, `text`, `title`, `date_begin`, `location_name`) ; il n’est pas désigné comme entrée par les chemins de configuration de la chaîne principale.

Ces chiffres décrivent les artefacts fournis, pas une collecte à une date certaine : aucune date de génération n’est établie par le seul contenu Parquet. Au **28 octobre 2026**, le corpus comporte **768 dates antérieures** à cette date et **65 dates égales ou postérieures**. Quatre fiches sont antérieures au 28 octobre 2025, borne de l’historique d’un an calculée à la date courante. Le corpus n’est donc pas une preuve de fraîcheur actuelle ; la soutenance doit distinguer ces fichiers figés d’une collecte en direct.

### Découpage (chunking)

Le traitement principal opère sur une fiche par événement. Aucun découpage systématique de longs textes en plusieurs segments n’est réalisé par `vectorize_events.py`. Le fichier séparé `chunks.parquet` ne suffit pas à établir qu’il s’agit de l’entrée réellement utilisée pour fabriquer l’index livré. La limite de longueur prévue dans les paramètres est de 512 tokens pour le modèle, mais le script de vectorisation ne montre pas de stratégie explicite de segmentation ni de contrôle de troncature.

### Embeddings : modèle, dimensions et batch

Le script de vectorisation encode des textes avec `BAAI/bge-small-en-v1.5`, en lots de **32**, convertit les vecteurs en `float32` et les ajoute à un `faiss.IndexFlatL2`. L’index fourni a été ouvert avec FAISS : il s’agit bien d’un `IndexFlatL2`, entraîné, avec **833 vecteurs de dimension 384**, et une métrique de distance L2.

La configuration déclare `EMBEDDING_DIM = 1024` et `EMBEDDING_NORMALIZE = True`. En revanche, le vectoriseur et le retriever spécifient tous deux `BAAI/bge-small-en-v1.5`, dont les embeddings ont 384 dimensions ; l’index fourni mesure également 384 dimensions. Le paramètre `EMBEDDING_DIM` n’est pas utilisé par ces chemins de vectorisation et de retrieval, et `encode` n’active pas explicitement la normalisation déclarée. Il s’agit donc d’une **incohérence de configuration** à clarifier, mais pas d’une panne certaine démontrée : les dimensions modèle/index concordent dans le chemin de code examiné. Le retriever ne contrôle toutefois pas explicitement cette compatibilité, ce qui serait utile en cas de changement de modèle. La dimension constatée de l’index fourni est 384, et non 1024.

Une autre différence de représentation apparaît dans le code : le nettoyage crée déjà un champ `text`, mais le script de vectorisation reconstruit son propre texte à partir du titre, des descriptions, des mots-clés, du lieu et de `date_begin`, puis remplace la colonne `text` dans les métadonnées enregistrées. Cette étape n’inclut pas exactement les mêmes éléments que le champ `text` construit par le nettoyage. Le texte du Parquet livré et le texte effectivement utilisé pour l’index ne peuvent donc pas être supposés identiques à partir des seuls chemins de configuration.

## 4. Choix du modèle NLP

### Modèles sélectionnés et rôle

Deux composants de modèle sont distincts :

- **Recherche sémantique** : `BAAI/bge-small-en-v1.5`, chargé localement par Sentence Transformers pour encoder les fiches et les questions ;
- **Génération** : `open-mistral-7b`, configuré par `ChatMistralAI` dans `mistral_qa.py`, avec une température de **0,2**. L’appel de génération utilise l’API Mistral et nécessite une clé fournie au runtime.

Le prompt du code indique la date du jour, demande une réponse en français, fournit le contexte récupéré, demande de signaler si la réponse n’y figure pas, interdit de mentionner un événement absent du contexte et indique de ne pas supposer de période temporelle non précisée par l’utilisateur. Il demande également de lister les événements à différentes dates sans filtrage additionnel.

### Évaluation et limites des modèles

Dans le script `evaluate_rag.py`, le modèle d’évaluation RAGAS est **Ollama `mistral`** par défaut ; les embeddings d’évaluation sont aussi locaux, via `HuggingFaceEmbeddings` et `BAAI/bge-small-en-v1.5`. Cela ne remplace pas le modèle Mistral utilisé par le chemin de génération RAG et ne signifie pas que le CSV a été produit sans dépendance à une configuration locale ou aux versions de bibliothèques en place.

Les limites à garder en vue sont la dépendance de la génération à l’API Mistral, à sa disponibilité et à ses conditions d’usage ; la dépendance du résultat à la qualité, à la couverture et à la fraîcheur des fiches ; et le risque qu’une récupération incomplète produise une réponse erronée ou un faux constat d’absence. Le modèle d’embedding indiqué dans le code est la variante `bge-small-en-v1.5` ; la performance sur des requêtes françaises n’est pas établie par les quelques scores fournis. Aucun benchmark de latence réelle, de coût ou de comparaison de modèles n’est documenté par les artefacts examinés.

## 5. Construction de la base vectorielle

### FAISS et persistance

Le code utilise un index FAISS exact `IndexFlatL2`, enregistré par `faiss.write_index`, et un fichier Parquet pour les métadonnées. Les chemins déclarés par `settings.py` sont :

```text
data/raw/events_raw.json
data/processed/events.parquet
data/vector_store/faiss_index.bin
data/vector_store/metadata.parquet
```

Les fichiers Parquet et l’index reçus dans l’espace de livraison se trouvent à la racine de cet espace, et non aux chemins `data/...` déclarés dans `settings.py`. Les données locales et les artefacts générés sont ignorés par `.gitignore` ; ils sont conçus comme résultats régénérables plutôt que comme fichiers à versionner systématiquement.

Le retriever vérifie que le champ `text` existe dans les métadonnées et que `index.ntotal == len(metadata)`. Cette égalité est vérifiée sur l’index/parquet fournis : **833 = 833**. Les métadonnées contiennent notamment `uid`, `title`, `location_name`, `date_begin` et le texte ; les objets retrouvés portent aussi une distance L2. Le code utilise le même modèle BGE Small pour indexer et encoder les requêtes, ce qui concorde avec les 384 dimensions de l’index fourni ; toutefois, le retriever ne vérifie pas explicitement cette dimension au runtime, et la seule égalité de cardinalité ne prouve pas l’alignement ligne-vecteur ni que le texte indexé reflète exactement celui du corpus source.

Le retriever calcule `k_effective = min(k, ntotal)` ; la génération demande `k=5` par défaut. Aucune valeur de seuil de distance n’est appliquée dans le code décrit, et le score L2 ne doit pas être interprété comme une probabilité de pertinence.

## 6. API et interface

### Framework et routes implémentées dans `main.py`

Le fichier `main.py` définit une application FastAPI avec les routes suivantes :

| Route | Rôle déclaré dans le code |
|---|---|
| `GET /health` | Retourne un état, la dernière reconstruction connue et la prochaine reconstruction planifiée. |
| `POST /ask` | Accepte un champ `query` non vide et retourne `{"response": "…"}`. |
| `POST /rebuild` | Programme le pipeline de collecte, nettoyage et vectorisation en tâche de fond. |
| `POST /rebuild/now` | Lance le pipeline immédiatement et attend son résultat. |
| `GET /rebuild/status` | Retourne l’état du verrou, les dates de reconstruction et l’état du scheduler. |

Le modèle Pydantic de requête est `QueryRequest` avec `query: str` et une longueur minimale de 1. La route `/ask` rejette aussi une chaîne composée d’espaces. Les erreurs inattendues de génération sont journalisées et transformées en HTTP 500 ; le détail de l’exception est inclus dans la réponse HTTP par le code actuel. Le scheduler déclare un rebuild hebdomadaire le lundi à 03:00, selon le fuseau horaire par défaut du processus.

La réponse de `/ask` ne renvoie que la chaîne `response` ; elle n’expose pas de liste de sources séparée dans le JSON. L’interface `app.py` est une application Streamlit qui envoie `{"query": …}` à `/ask`, à l’URL par défaut `http://localhost:8000` (surcharge possible par `PULS_API_URL`), et accepte plusieurs noms de champ de réponse.

### Exemple de requête au contrat du code

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

### Lancement déclaré et limites de reproductibilité

Le Makefile déclare `make api` (`uvicorn api.main:app --reload`) et `make ui` (`streamlit run ui/app.py`). Les modules et pièces jointes reçus sont présentés à plat, alors que les imports et commandes décrivent une organisation en packages (`api/`, `config/`, `scripts/`, `ui/`, ainsi que `tests/`). L’aplatissement des pièces jointes ne permet pas de conclure que ces dossiers manquent au dépôt source ; il empêche simplement de valider leur présence et les chemins exécutables à partir de cette seule représentation. Aucune commande de lancement n’a été exécutée avec succès ni revendiquée.

## 7. Évaluation du système

### Jeu de questions annotées et schéma

Le jeu annoté de priorité utilisé pour ce README est `annotated_qa(2).json`, qui contient **42 questions uniques**. Chaque entrée est un objet avec les trois clés `question`, `ground_truth` et `reference_contexts` ; les champs de référence sont renseignés pour les 42 entrées. Le plus petit `annotated_qa.json` contient **3 questions**, avec le même schéma. `annotated_qa(1).json` contient ces trois questions mais seulement la clé `question` : il ne comporte ni réponse de référence ni contextes de référence. Ces variantes ne doivent pas être confondues.

Le CSV livré contient **3 lignes**, avec les colonnes `user_input`, `retrieved_contexts`, `reference_contexts`, `response`, `reference`, `faithfulness`, `answer_relevancy` et `context_precision`. Ses trois questions correspondent exactement aux trois questions du petit jeu `annotated_qa.json` et ne sont pas présentes dans le jeu de 42 questions (`annotated_qa(2).json`). Le script `evaluate_rag.py` à la racine limite l’évaluation à **3** entrées ; une autre variante du script téléversée indique une limite de 42, mais ce n’est pas celle qui correspond au petit CSV selon ses questions. Le CSV ne documente pas, à lui seul, la commande, les versions de dépendances, la configuration Ollama ou la date d’exécution ayant produit les scores.

Le schéma attendu par le script comprend une question, éventuellement `ground_truth` et `reference_contexts`, des contextes retrouvés et une réponse générée. Le code adapte le nommage des champs aux versions de RAGAS historiques ou récentes. La présence d’un champ de référence ne décrit pas à elle seule le protocole d’annotation ; l’échantillon demeure très réduit et ne représente pas toutes les requêtes possibles.

### Métriques et résultats du CSV fourni

Les moyennes ci-dessous ont été recalculées à partir des trois lignes du fichier CSV, sans arrondir les valeurs avant le calcul :

| Métrique | Valeurs par cas | Moyenne (n = 3) |
|---|---|---:|
| `faithfulness` | 1,000 ; 0,400 ; 1,000 | **0,8000** |
| `answer_relevancy` | 0,848167 ; 0,844886 ; 0,908151 | **0,8671** |
| `context_precision` | 0,450 ; 1,000 ; 0,000 | **0,4833** |

Les trois résultats illustrent des comportements contrastés :

- **Rap/pop à La Vapeur, novembre–décembre 2026** : `faithfulness` 1,0, `answer_relevancy` 0,8482, `context_precision` 0,45. La réponse est rédigée à partir du contexte, mais le score de précision de contexte est intermédiaire.
- **Thylacine à La Vapeur en 2027** : `faithfulness` 0,4, `answer_relevancy` 0,8449, `context_precision` presque 1,0. Une bonne pertinence des contextes récupérés ne suffit donc pas à garantir une réponse fidèle à la référence.
- **Jazz à La Vapeur en novembre 2026** : `faithfulness` 1,0, `answer_relevancy` 0,9082, `context_precision` 0,0. La réponse du CSV affirme qu’aucun concert de jazz n’est prévu, alors que la référence en mentionne deux. Le score de fidélité au contexte ne signifie pas ici que la conclusion est correcte par rapport à la référence : le cas illustre un défaut de couverture/récupération et un faux négatif.

Le CSV ne contient pas de colonne `context_recall` ; aucun score ni moyenne de rappel de contexte ne peut donc être rapporté à partir de ce fichier. Les métriques RAGAS ne constituent pas une mesure directe de satisfaction utilisateur, et trois observations ne permettent pas de généraliser à la qualité globale du système. Aucun nouveau calcul d’évaluation n’a été exécuté pour ce README.

## 8. Recommandations et perspectives

### Points effectivement observables

- Une chaîne de traitement de bout en bout est représentée dans les scripts : collecte, nettoyage, vectorisation, recherche et génération.
- Les fichiers fournis permettent d’inspecter un corpus de 833 fiches, des métadonnées correspondantes et un index FAISS de 833 vecteurs.
- Le jeu annoté de 42 cas possède des réponses et contextes de référence, et le prompt demande explicitement de ne pas inventer d’événement absent du contexte.
- L’API et l’interface décrivent des contrats simples et des routes de statut/reconstruction.

Ces constats décrivent les fichiers examinés ; ils ne remplacent pas l’exécution reproductible de l’application et des tests.

### Limites établies par les artefacts

- **Fraîcheur et périmètre temporel :** la date exacte de production des Parquet n’est pas établie. Au 28 octobre 2026, 768 des 833 dates d’événements sont passées, et seulement 65 sont à venir ou datées du jour. Une collecte fraîche n’a pas été faite dans le cadre de la vérification.
- **Embeddings/index :** la configuration déclare `EMBEDDING_DIM = 1024`, mais le modèle BGE Small utilisé dans le code et l’index fourni sont en 384 dimensions. Le paramètre déclaré n’est pas consommé dans les chemins inspectés ; puisque modèle d’indexation, modèle de requête et index concordent en 384, les éléments disponibles ne démontrent pas une panne certaine. La normalisation déclarée n’est pas explicitement passée à `encode`, et le retriever ne contrôle pas la dimension avant recherche.
- **Représentation du texte :** le texte utilisé par le code de vectorisation est reconstruit différemment de celui produit lors du nettoyage. Le corpus principal et `chunks.parquet` ne sont pas interchangeables sans établir leur provenance.
- **Évaluation :** les scores disponibles concernent trois questions seulement. Un cas de jazz montre un faux négatif par rapport à la réponse de référence. Le CSV ne comprend ni `context_recall` ni mesure de satisfaction, et ne permet pas de mesurer la fraîcheur, la latence ou le coût.
- **Structure attendue :** les imports, la configuration setuptools, le Makefile et le Dockerfile décrivent des packages et répertoires dédiés. Les pièces jointes examinées étant aplaties, leur présentation ne permet pas de conclure que le dépôt source manque de cette structure ; l’arborescence cible est explicitée en section 9.
- **Docker :** le Dockerfile prévoit de copier `api/`, `config/`, `scripts/` et `ui/`. L’aplatissement des pièces jointes ne permet pas d’affirmer que ces répertoires manquent au dépôt source ; aucune construction ne permet ici de vérifier le contexte réel. Son étape d’installation copie `pyproject.toml` avant l’installation ; le `.dockerignore` exclut `.env`, `data/`, les tests et le CSV RAGAS du contexte d’image, et le modèle d’embedding est téléchargé pendant la construction. Aucun `docker build` ni lancement de conteneur n’a été exécuté : le fonctionnement de l’image et la disponibilité des artefacts au runtime ne sont donc pas établis.
- **Reproductibilité des dépendances :** `mistral_qa.py` importe `truststore`, absent de la liste `core` du `pyproject.toml` principal (il apparaît dans une autre variante téléversée). Les variantes de fichiers présentes ne sont pas toutes identiques ; il convient de rattacher tout résultat à la version effectivement exécutée, sans déduire sa provenance du seul nom.
- **Réseau et secrets :** la génération dépend d’une clé Mistral ; la collecte dépend d’une clé OpenAgenda. Aucun secret réel ne doit figurer dans le dépôt ou dans le README. La collecte désactive la vérification TLS (`verify=False`) dans la requête HTTP, ce qui est un point de vigilance de sécurité visible dans le code.

### Perspectives

Pour une suite du projet, les axes de travail sont de stabiliser la structure et les versions des composants, de documenter la provenance et la date des corpus, de consolider la cohérence entre texte préparé, embeddings et index, et d’élargir l’évaluation à des cas positifs, négatifs, ambigus et hors périmètre. Il serait également utile d’observer séparément la couverture des résultats, la fidélité factuelle, la latence et le coût, puis de rendre les reconstructions vérifiables et sûres en déploiement.

Ces éléments sont présentés comme **limites et perspectives** pour cadrer le POC ; ce README ne prétend pas qu’ils ont été réalisés, ni qu’un test ou une commande de déploiement a réussi.

## 9. Organisation du dépôt et commandes

### Arborescence attendue selon les imports et les commandes

Les pièces jointes fournies sont représentées sous forme aplatie. Cela décrit leur mode de transmission et ne prouve ni la présence ni l’absence des répertoires dans le dépôt source. Selon les imports, `pyproject.toml`, le Makefile et le Dockerfile, l’arborescence attendue est :

```text
project/
├── api/                 # main.py, faiss_retriever.py, mistral_qa.py
├── config/              # settings.py, geo.py
├── scripts/             # fetch_events.py, clean_events.py, vectorize_events.py, evaluate_rag.py
├── tests/               # test_api.py, test_pipeline.py
├── ui/                  # app.py
├── data/                # artefacts locaux/régénérables (raw/, processed/, vector_store/)
├── pyproject.toml
├── Makefile
└── Dockerfile
```

Le tableau suivant inventorie les fichiers utiles effectivement transmis, sans en déduire l’état complet de l’arborescence du dépôt :

| Fichier livré | Rôle repéré dans le code |
|---|---|
| `fetch_events.py` | Collecte OpenAgenda et écriture du JSON brut. |
| `clean_events.py` | Aplatissement, nettoyage, filtres et génération du corpus préparé. |
| `vectorize_events.py` | Encodage des textes, index FAISS et écriture des métadonnées. |
| `faiss_retriever.py` | Chargement de l’index et recherche des documents. |
| `mistral_qa.py` | Prompt LangChain et génération Mistral. |
| `main.py` | Définition de l’API FastAPI et des tâches planifiées. |
| `app.py` | Interface Streamlit cliente de `/ask`. |
| `evaluate_rag.py` | Génération de réponses et évaluation RAGAS. |
| `test_api.py`, `test_pipeline.py` | Tests écrits pour l’API et le pipeline. |
| `settings.py`, `geo.py` | Paramètres et fonctions de filtrage géographique. |
| `pyproject.toml`, `Makefile`, `Dockerfile` | Dépendances, commandes et recette de conteneur déclarées. |
| `events.parquet`, `metadata.parquet`, `chunks.parquet` | Données fournies pour inspection. |
| `faiss_index.bin` | Index fourni, 833 vecteurs, 384 dimensions. |
| `annotated_qa.json`, `annotated_qa(2).json` | Jeux annotés de 3 et 42 entrées respectivement. |
| `ragas_evaluation_results.csv` | Scores RAGAS fournis pour trois entrées. |

`README(1).md`, `README(2).md`, des fichiers suffixés par `(1)` ou `(2)` et plusieurs versions de configuration/script sont également présents dans l’espace d’upload. Les doublons ne sont pas tous de contenu identique : par exemple, une variante de l’évaluation fixe la limite à 42 au lieu de 3, et une variante de `pyproject.toml` ajoute `truststore`. Les statistiques et descriptions de ce README utilisent en priorité les fichiers sans suffixe pour le code/configuration, et le jeu de 42 questions `annotated_qa(2).json` pour décrire le jeu annoté définitif ; le CSV livré reste évalué séparément sur ses trois questions.

### Installation déclarée

Le projet déclare Python **3.10 ou supérieur**. Depuis une copie complète et structurée du projet, les commandes d’installation inscrites au Makefile sont :

```bash
python -m venv .venv
source .venv/bin/activate
make install       # pip install -e ".[core]"
make install-dev  # ajoute les dépendances de développement/évaluation
make install-ui   # ajoute Streamlit et requests
```

Sous Windows, le Makefile indique l’usage de WSL, Git Bash ou des commandes pip équivalentes. Pour exécuter la collecte, il faut configurer localement les variables requises par le service, notamment `OPENAGENDA_API_KEY` et, pour interroger le générateur, `MISTRAL_API_KEY`. Les valeurs sont à fournir à l’environnement d’exécution ; **aucune valeur de clé ne doit être inscrite dans un fichier partagé**. La création d’un fichier `.env` local est une possibilité, mais aucun secret de démonstration n’est fourni ici.

### Commandes inscrites au Makefile et état de vérification

Les commandes ci-dessous sont celles déclarées par le Makefile ; elles ne sont pas présentées comme ayant été exécutées avec succès :

| Cible | Commande inscrite | Observation statique dans l’arborescence livrée |
|---|---|---|
| `make fetch` | `python -m scripts.fetch_events` | Le Makefile attend le package `scripts`. Les pièces jointes aplaties ne permettent pas de vérifier ce chemin dans le dépôt source. Requiert également OpenAgenda. |
| `make clean-data` | `python -m scripts.clean_events` | Le chemin de module attendu est `scripts.clean_events` ; non vérifiable à partir des pièces jointes aplaties. |
| `make vectorize` | `python -m scripts.vectorize_events` | Le chemin de module attendu est `scripts.vectorize_events` ; non vérifiable à partir des pièces jointes aplaties. Requiert modèle, FAISS, Parquet et données. |
| `make pipeline` | `fetch`, puis `clean-data`, puis `vectorize` | Enchaîne les trois étapes précédentes ; la collecte est une opération réseau. |
| `make api` | `uvicorn api.main:app --reload` | Attend `api.main:app` ; la présence de ce chemin dans le dépôt n’est pas vérifiable avec les pièces jointes aplaties. |
| `make ui` | `streamlit run ui/app.py` | Attend `ui/app.py` ; la présence de ce chemin dans le dépôt n’est pas vérifiable avec les pièces jointes aplaties. |
| `make test` | `pytest tests/ -v` | Attend `tests/` ; les fichiers de test transmis étant à plat, leur emplacement dans le dépôt n’est pas établi. Aucun résultat de test n’est revendiqué. |
| `make evaluate` | `python -m tests.evaluate_rag` | Le chemin de module ne correspond pas à `evaluate_rag.py` à la racine ; l’aide du Makefile affiche une autre cible (`scripts.evaluate_rag`). |
| `make docker-build` | `docker build -t puls-events-rag .` | Le Dockerfile attend les répertoires qu’il copie ; la représentation aplatie ne permet pas d’en vérifier le contexte source. Aucune construction n’a été exécutée. |
| `make docker-run` | `docker run -p 8000:8000 --env-file .env -v $(PWD)/data:/app/data puls-events-rag` | Requiert une image construite et un environnement runtime. |

La commande `make all` enchaîne le pipeline, les tests et l’évaluation ; elle n’est pas un contrôle hors ligne et n’a pas été exécutée. Le Makefile comprend aussi `make clean`, qui supprime les artefacts sous `data/vector_store/` : ne pas l’utiliser si l’on souhaite conserver un index local.

## 10. Annexes

### Exemple de structure d’une entrée annotée

Le schéma constaté dans les fichiers annotés complets est :

```json
{
  "question": "Quels concerts de jazz sont prévus à La Vapeur en novembre 2026 ?",
  "ground_truth": "Deux concerts de jazz sont prévus…",
  "reference_contexts": ["Extrait de la fiche d’événement…"]
}
```

Le fichier de 42 entrées utilise les mêmes trois champs. Le JSON de 3 questions les utilise également ; la variante de 3 questions ne contenant que `question` ne doit pas être prise pour le benchmark complet.

### Exemple de prompt et de réponse API

Le prompt réel est défini dans `mistral_qa.py` et comprend une date, une consigne de réponse en français, le contexte récupéré et la question. Son principe est de répondre à partir des seuls extraits et de signaler quand l’information n’y figure pas. Le résultat HTTP renvoyé par `/ask` est de la forme `{"response": "…"}` ; le code n’inclut pas de tableau de citations indépendant dans cette réponse.

### Vérifications effectuées pour ce README

- Extraction du texte et de la structure du template DOCX, puis organisation du README selon ses dix sections attendues.
- Lecture statique du `Makefile`, `Dockerfile`, `.dockerignore`, `.gitignore`, `pyproject.toml` et des scripts de collecte, nettoyage, géographie, vectorisation, retrieval, génération, API, interface et évaluation.
- Analyse des trois variantes du jeu annoté et du schéma du CSV d’évaluation ; recomputation des moyennes des métriques disponibles.
- Lecture des schémas et contenus de `events.parquet`, `metadata.parquet` et `chunks.parquet` ; vérification de l’égalité ligne à ligne des colonnes communes entre `events.parquet` et `metadata.parquet`.
- Ouverture de l’index fourni avec FAISS et constat de son type, de sa dimension et de son nombre de vecteurs.
- Comparaison des chemins réellement présents avec ceux attendus par les imports, commandes et instructions Docker.

Aucun appel API, aucune collecte, aucun appel Mistral/Ollama, aucune nouvelle évaluation RAGAS, aucun test pytest et aucune construction ou exécution Docker n’ont été réalisés pour rédiger ce document.
