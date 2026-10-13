# Puls-Events Dijon — assistant RAG de recommandations culturelles

## 1. Objectifs du projet

### Contexte

La mission consiste à construire un assistant intelligent pour Puls-Events, capable de recommander en français des événements culturels autour de Dijon et de la Côte-d’Or à partir des données d’OpenAgenda.

### Problématique

Un système RAG (Retrieval-Augmented Generation) doit permettre de rechercher les événements pertinents dans une base actualisable, puis de fournir une réponse contextualisée par un modèle de langage. L’enjeu métier est de limiter les réponses non fondées tout en facilitant l’accès aux événements par question en langage naturel.

### Objectif du POC

Le POC cherche à démontrer :

- la faisabilité technique de la chaîne collecte → nettoyage → vectorisation → recherche → génération ;
- la valeur métier d’une recherche sémantique d’événements culturels ;
- la possibilité d’exposer cette chaîne via une API et de l’évaluer sur un jeu de questions annotées.

L’état livré doit toutefois être présenté comme un POC architectural à finaliser, et non comme un service industrialisé démontré de bout en bout.

### Périmètre

Le périmètre couvre les événements culturels autour de Dijon et de la Côte-d’Or, collectés depuis OpenAgenda. La présentation et le code décrivent les routes `/ask` et `/rebuild`; le code livré ajoute également `/health`. Les données et artefacts générés localement ne doivent pas être versionnés.

## 2. Architecture du système

### Schéma global (schéma UML)

```text
OpenAgenda
   │ clé d’environnement
   ▼
collecte → nettoyage → embeddings BGE → index FAISS
                                      │
question → recherche sémantique ─────┘
                                      ▼
                              contexte → Mistral → réponse

annotated_qa.json → évaluation Ragas → ragas_evaluation_results.csv
```

### Données entrantes (API Open Agenda)

La collecte OpenAgenda gère la pagination, les tentatives répétées, un fallback par agendas et un filtre géographique. La clé d’API doit être fournie par variable d’environnement et ne doit jamais être écrite dans le dépôt, l’image Docker ou les logs.

### Prétraitement / embeddings / base vectorielle

Les événements sont nettoyés (HTML, dates UTC et champ `text`), puis vectorisés avec `BAAI/bge-small-en-v1.5`. Les vecteurs sont stockés dans un index FAISS CPU et associés à leurs métadonnées. Le pipeline doit garantir une représentation canonique du texte et vérifier `faiss.ntotal == len(metadata)`.

### Intégration LLM avec LangChain

Le retriever récupère les documents les plus proches et transmet leur contexte à Mistral via LangChain. Le prompt doit imposer une réponse fondée sur les documents récupérés et un comportement explicite lorsqu’aucun événement pertinent n’est disponible.

### Exposition via API

L’API cible est une application FastAPI servie par `uvicorn api.main:app`. L’arborescence actuelle ne fournit toutefois pas `api/main.py` ni les packages `api`, `config` et `scripts` aux emplacements attendus par certains imports : cette intégration doit être corrigée et vérifiée dans un environnement vierge.

### Technologies utilisées

- Python, FastAPI et Uvicorn ;
- OpenAgenda pour la collecte ;
- nettoyage Python des événements ;
- `sentence-transformers` / modèle `BAAI/bge-small-en-v1.5` pour les embeddings ;
- FAISS CPU pour la recherche vectorielle ;
- LangChain et Mistral pour la génération ;
- Ragas pour l’évaluation ;
- Docker pour le déploiement ;
- pytest pour les tests à rendre autonomes.

## 3. Préparation et vectorisation des données

### Source de données : API Open Agenda (paramètres utilisés, filtres appliqués)

La source est l’API OpenAgenda. La collecte prévoit pagination, retry, fallback par agendas et filtrage géographique. Les paramètres effectivement utilisés doivent être consignés avec la date de collecte, la zone, les agendas interrogés et les limites de pagination afin de rendre le benchmark reproductible.

### Nettoyage : Exemples d’anomalies corrigées, méthodes utilisées

Le script de nettoyage traite notamment le HTML, normalise les dates en UTC et construit un champ `text` destiné à la recherche. Les tests doivent couvrir les dates invalides, les doublons, les champs absents et les événements sans contenu exploitable.

### Chunking : Raison du découpage, taille choisie

Le POC travaille sur un document par événement plutôt que sur un long corpus découpé arbitrairement : le titre, le lieu, la date, la description et les informations utiles sont regroupés dans le champ `text`. La taille et la stratégie devront être documentées si des événements longs nécessitent un découpage en plusieurs chunks.

### Embedding

#### Modèle utilisé

Le code livré mentionne `BAAI/bge-small-en-v1.5`, exécuté localement pour produire les embeddings. Cette information doit être figée avec la version de la librairie et le modèle exact utilisés lors de l’évaluation.

#### Dimensionnalité, logique de batch, format des vecteurs

La dimension, la taille des batches, le type numérique et la normalisation doivent être consignés dans les métadonnées de l’index. Les embeddings sont normalisés, tandis que l’index utilisé est `IndexFlatL2` : ce choix de métrique doit être explicité ou remplacé par une métrique cohérente avec le protocole de recherche.

Une incohérence reste à corriger : `vectorize_events.py` reconstruit le texte à partir de colonnes alors que le nettoyage produit déjà `text`. Une seule représentation canonique doit être retenue.

## 4. Choix du modèle NLP

### Modèle sélectionné

Mistral est utilisé pour générer la réponse à partir du contexte récupéré. Les embeddings reposent séparément sur `BAAI/bge-small-en-v1.5`.

### Pourquoi ce modèle ? (Critères : coût, qualité, compatibilité LangChain…)

Mistral est compatible avec l’intégration LangChain et convient à un POC de questions-réponses en français. Le choix doit être évalué avec les versions, le coût, la latence, les quotas et la qualité observée sur un benchmark suffisamment large.

### Prompting (si utilisé) : Prompt de base / structure

Le prompt doit recevoir la question utilisateur et les événements récupérés, demander une réponse en français et interdire l’invention de faits absents du contexte. Il doit également prévoir le cas où le retriever ne trouve aucun document fiable. Le prompt exact utilisé doit être ajouté en annexe et versionné si sa formulation influence les résultats.

### Limites du modèle

Les limites comprennent la dépendance à la disponibilité et aux quotas de Mistral, le risque d’hallucination, la sensibilité à la qualité et à la fraîcheur des données, ainsi que l’absence démontrée de mémoire conversationnelle. Les réponses doivent être contrôlées sur les dates, les lieux, les sources et les événements hors périmètre.

## 5. Construction de la base vectorielle

### Faiss utilisé

FAISS CPU est utilisé pour la recherche vectorielle. Le retriever vérifie la présence du champ `text` et l’alignement entre l’index et les métadonnées. Le choix `IndexFlatL2` doit être documenté au regard de la normalisation des embeddings.

### Stratégie de persistance

L’index et ses métadonnées sont des artefacts régénérables à partir des données collectées et nettoyées. Ils doivent être reconstruits par script, remplacés de manière atomique et versionnés logiquement avec le modèle d’embedding, la dimension, la métrique, `k`, la date de collecte et le nombre de documents.

### Format de sauvegarde ?

Le format attendu est un fichier d’index FAISS accompagné d’un fichier de métadonnées sérialisées. Les chemins exacts et le schéma doivent être stabilisés dans la configuration et documentés avec la commande de reconstruction.

### nommage ?

Le nommage des artefacts doit distinguer l’index, les métadonnées et les sorties d’évaluation. Les fichiers générés sous `data/` et `ragas_evaluation_results.csv` sont exclus du contrôle de version par `.gitignore`.

### Métadonnées associées

#### Ce qui est conservé pour chaque document

Chaque événement doit conserver au minimum son texte de recherche et les informations permettant de justifier la réponse : titre, lieu, date, URL si disponible et identifiant source. L’alignement entre chaque ligne de métadonnées et sa position FAISS doit être vérifié à chaque reconstruction.

## 6. API et endpoints exposés

### Framework utilisé : FastAPI / Flask

Le framework ciblé est FastAPI. La commande de lancement attendue est :

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

### Endpoints clés

#### `/ask` : question utilisateur, réponse du système

Reçoit une question en français, recherche les événements pertinents et demande à Mistral de générer une réponse à partir du contexte récupéré.

#### `/rebuild` : reconstruction de l’index (si besoin)

Déclenche la reconstruction de l’index. Dans l’état actuel, la tâche de fond ne fournit pas encore d’état observable ni de verrou de concurrence ; le contrat HTTP, le suivi d’état et le remplacement atomique restent à définir.

### Format des requêtes/réponses

Le contrat doit documenter le schéma JSON de la question, la réponse générée, les sources retournées, les erreurs et les codes HTTP. Une réponse devrait exposer les documents récupérés afin de permettre la vérification des dates, lieux et URLs.

### Exemple d’appel API : avec curl ou code Python

```bash
curl http://localhost:8000/health
curl -X POST http://localhost:8000/ask \
  -H 'Content-Type: application/json' \
  -d '{"query":"Quels sont les prochains concerts à La Vapeur ?"}'
curl -X POST http://localhost:8000/rebuild
```

### Tests effectués et documentés

Les fichiers de tests présents sont `test_api.py` et `test_pipeline.py`, mais ils importent `api.main`, dépendent potentiellement d’artefacts `data/` absents et peuvent appeler OpenAgenda ou Mistral. Ils doivent être isolés par mocks et fixtures, puis exécutés avec `pytest -q` dans un environnement vierge.

### Gestion des erreurs / limitations

Les erreurs internes ne doivent pas être renvoyées avec leur détail au client. Il faut traiter les clés absentes, l’index absent ou vide, `k > ntotal`, les dimensions incohérentes, les réponses sans résultat et les échecs des services externes. CORS, utilisateur Docker non root et healthcheck restent à finaliser.

## 7. Évaluation du système

### Jeu de test annoté

#### Nombre d’exemples

`annotated_qa.json` est mentionné par le projet mais est exclu du contrôle de version par `.gitignore`. Le CSV d’évaluation présent dans l’ancien audit contient 3 lignes d’évaluation et 8 colonnes.

#### Méthode d’annotation

Les questions, contextes récupérés, réponses et références sont présents pour les trois cas du CSV. La méthode d’annotation, les critères de référence et le lien avec les événements sources doivent être explicités pour permettre une reproduction indépendante.

### Métriques d’évaluation

#### Exemple : similarité sémantique, taux de couverture des réponses, score de satisfaction (subjectif)

Les métriques observées sont `faithfulness`, `context_precision`, `context_recall` et `answer_relevancy`, avec un complément d’analyse humaine de l’exactitude des dates, lieux, sources et de l’absence d’hallucination.

### Résultats obtenus

#### Analyse quantitative (scores globaux)

Le CSV corrigé rapporté par l’audit contient :

- `faithfulness` : **1,000 ; 0,750 ; 0,875**, moyenne simple **0,875** ;
- `context_precision` : **0,000 ; 1,000 ; 0,000**, moyenne **0,333** ;
- `context_recall` : **0,500 ; 1,000 ; 0,000**, moyenne **0,500** ;
- `answer_relevancy` : **NaN sur les 3 lignes**, donc aucune moyenne exploitable. Ce n’est pas un score nul.

Trois questions ne suffisent pas pour conclure à la performance générale. Le benchmark doit inclure des cas positifs, négatifs, ambigus, hors périmètre, sans résultat, avec dates et lieux variés.

#### Analyse qualitative (exemples de bonnes/mauvaises réponses)

La présentation d’exemples doit montrer une réponse correctement sourcée, un cas sans résultat et un cas d’erreur ou de date ambiguë. Les exemples doivent être rejoués après correction de l’arborescence et accompagnés du modèle, de la version Ragas, du modèle d’embeddings, de `k`, de la date de collecte et de la commande de production du CSV.

## 8. Recommandations et perspectives

### Ce qui fonctionne bien

- la chaîne métier cible est clairement identifiée ;
- la collecte OpenAgenda prévoit pagination, retry, fallback et filtrage géographique ;
- le nettoyage, la vectorisation BGE, FAISS, LangChain et Mistral sont représentés dans les fichiers livrés ;
- la présentation couvre le besoin, l’architecture, les données, la recherche, la génération, l’API, l’évaluation et les perspectives.

### Limites du POC

#### Volumétrie, performance, coût, couverture thématique ?

La volumétrie évaluée est trop faible pour caractériser la performance : seulement trois questions figurent dans le CSV rapporté. La latence, le coût Mistral, la couverture thématique, la fraîcheur OpenAgenda, les quotas et le comportement de FAISS local/non partagé ne sont pas démontrés. La structure des packages, les dépendances et les artefacts générés ne sont pas encore reproductibles sans correction.

### Améliorations possibles

#### Ajout de… / Amélioration de..

- restaurer `config/settings.py` et `config/geo.py` et figer les dépendances ;
- corriger l’arborescence et les imports, puis vérifier `pip install -e .` et `docker build` ;
- rendre les tests indépendants du réseau, de Mistral et d’un index persistant ;
- définir un texte canonique, une métrique FAISS cohérente et un contrôle d’alignement ;
- enrichir le benchmark et recalculer le CSV, notamment `answer_relevancy` ;
- ajouter les sources dans les réponses et documenter les erreurs.

#### Passage en production via…

Le passage en production nécessite une CI/CD, une image non root, un healthcheck, des secrets injectés à l’exécution, CORS restreint, une reconstruction atomique avec verrou, un état observable, une mise à jour planifiée, un stockage vectoriel partagé et du monitoring.

## 9.  Organisation du dépôt GitHub

### Arborescence du dépôt (fichiers clés, scripts, dossiers)

L’inventaire réel fourni contient les fichiers suivants. L’arborescence ci-dessous ne reproduit que les éléments non exclus par `.gitignore`; les noms `README(1).md`, `README(2).md` et `Template...` correspondent aux fichiers présents dans l’espace fourni, pas à une structure Python déjà corrigée.

```text
.
├── .gitignore
├── README.md
├── README(1).md
├── README(2).md
├── POC-dun-Assistant-Intelligent-de-Recommandations-Culturelles.pptx
├── clean_events.py
├── evaluate_rag.py
├── faiss_retriever.py
├── fetch_events.py
├── image.png
├── image(1).png
├── mistral_qa.py
├── test_api.py
├── test_pipeline.py
└── vectorize_events.py
```

Le dépôt livré est donc plat. Le README historique proposait une cible `api/`, `config/`, `scripts/` et `tests/`, mais ces répertoires ne sont pas présents dans l’inventaire actuel. `Dockerfile`, `annotated_qa.json`, les fichiers PDF et DOCX, le fichier `ragas_evaluation_results.csv` s’il est généré, `data/`, les environnements virtuels, les caches, les logs, les fichiers IDE et les fichiers temporaires sont exclus par `.gitignore` et ne figurent pas dans l’arborescence versionnée.

### Explication rapide de chaque répertoire

Aucun répertoire applicatif n’est présent dans la livraison actuelle. Les scripts Python à la racine correspondent aux étapes suivantes :

- `fetch_events.py` : collecte des événements OpenAgenda ;
- `clean_events.py` : nettoyage et préparation des événements ;
- `vectorize_events.py` : génération des embeddings et construction de l’index ;
- `faiss_retriever.py` : recherche dans l’index FAISS ;
- `mistral_qa.py` : génération de réponses via Mistral ;
- `evaluate_rag.py` : évaluation du pipeline ;
- `test_api.py` et `test_pipeline.py` : tests à rendre autonomes ;
- `image.png`, `image(1).png` : images présentes dans l’espace fourni ;
- le fichier `.pptx` : présentation du POC ;
- les trois README : documents textuels présents, dont le présent README généré.

Pour une organisation GitHub maintenable, il est recommandé de déplacer l’application vers `api/`, les réglages vers `config/`, les scripts vers `scripts/` et les tests vers `tests/`, en ajoutant les `__init__.py` nécessaires. `src/` est explicitement ignoré par `.gitignore` et ne doit donc pas être utilisé sans réviser cette règle.

## 10. Annexes

### Extraits du jeu de test annoté

Le fichier `annotated_qa.json` est signalé par l’ancien README mais est exclu par `.gitignore`; aucun extrait ne doit être présenté comme versionné sans vérifier sa présence et ses règles de diffusion. Le CSV d’évaluation rapporté contient des questions, contextes récupérés, réponses et références pour trois cas.

### Prompt utilisé

Le prompt doit être fourni avec la version du modèle et du retriever. Il doit demander une réponse française fondée exclusivement sur le contexte, citer les informations utiles sur les événements et signaler l’absence de résultat plutôt que d’inventer une recommandation.

### Extraits de logs ou exemples de réponse JSON

Les logs contenant des secrets, des URLs sensibles ou des détails d’exception doivent rester exclus (`*.log` est ignoré). Pour la soutenance, fournir des logs nettoyés et un exemple JSON non sensible montrant la question, la réponse, les sources récupérées et, si nécessaire, le statut de reconstruction.
