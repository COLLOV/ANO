# AGENT.MD - Règles générales pour l'assistant codex

1. **No artifacts** - Éviter la création d'artefacts inutiles.

2. **Less code is better than more code** 

3. **No fallback mechanisms** — they hide real failures - Les mécanismes de secours masquent les vraies défaillances.

4. **Rewrite existing components over adding new ones** - Réécrire les composants existants plutôt que d'en ajouter de nouveaux.

5. **Flag obsolete files to keep the codebase lightweight** - Signaler les fichiers obsolètes pour maintenir une base de code légère.

6. **Avoid race conditions at all costs** - Éviter les conditions de concurrence à tout prix.

7. **Take your time to ultrathink when on extended thinking mode** — thinking is cheaper than fixing bugs - Prendre le temps de bien réfléchir en mode de pensée étendue, la réflexion coûte moins cher que la correction de bugs.

8. **Coder de façon modulaire pour favoriser la collaboration entre agents** - IMPORTANT : L'ajout de features ne doit pas casser le reste du système.

9. **Add comments only when necessary** — the code should speak for itself - Ajouter des commentaires uniquement quand nécessaire, le code doit être auto-explicatif.

10. **Always add meaningful logs** — but only where it brings value - Toujours ajouter des logs significatifs, mais seulement là où cela apporte de la valeur.

11. **Toujours penser production** le code doit toujours etre prêt pour la production. Pas de code inutiles. 

12. **Toujours mettre à jour le README.md** : Permet de documenter les changements.  

13. **Toujours creer un mode local et un mode api** : pour la gestion des llm, le mode local utilisera VLLM alors que le mode api utilisera une clé api d'un provider externe.


Ces règles visent à maintenir une base de code propre, modulaire et maintenable tout en favorisant la collaboration efficace entre différents agents et développeurs.
Utilise toujours uv pour gestion de package python
Don't write/change any code until you're very confident (95% or more) in what needs to be done. If unclear, ask for more info.

A chaque fois que tu fais une modif, commit les changements