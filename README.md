# VolcanEye

> **Projet pédagogique.** VolcanEye est un projet d'apprentissage en data science. Ce n'est **pas** un outil de surveillance ni d'alerte volcanique. Ses résultats ne doivent pas servir à prendre des décisions de sécurité. Pour l'information officielle sur le Piton de la Fournaise, consultez l'[OVPF-IPGP](https://www.ipgp.fr/observation/ovpf/).

## Objectif

Estimer, à partir de données sismiques publiques, la probabilité qu'un volcan entre en éruption dans les 8 heures. Le volcan de référence est le Piton de la Fournaise (La Réunion).

## État actuel

- **Données :** 609 éruptions sur 143 volcans. Un relevé des canaux montre que 40 % de ces éruptions ont une station qui renvoie un signal sismique exploitable.
- **Pipeline :** des workflows GitHub Actions (déclenchement manuel) téléchargent des fenêtres sismiques avec et sans éruption, en calculent les features, puis entraînent un modèle GRU. Les résultats sont stockés sur Google Cloud Storage.
- **Modèle :** il distingue une fenêtre avec éruption d'une fenêtre calme, mais ne fait pas encore de prévision.
- **API :** le modèle n'y est pas encore branché ; seul le « replay » de dates passées fonctionne.

## Contenu du dépôt

| Dossier | Rôle |
|---|---|
| `api/` | Service FastAPI, avec une interface Gradio sur `/ui` |
| `scripts/` | Téléchargement des fenêtres, relevé des canaux, features, entraînement, chargement de la base |
| `.github/workflows/` | Exécution de ces scripts sur GitHub Actions |
| `notebooks/` | Exploration et traitement du signal (ObsPy) ; `notebooks/utils` est utilisé par les scripts |
| `ml_logic/` | Premier modèle de référence, suivi avec MLflow |
| `supabase/` | Schéma de la base (volcans, éruptions, stations) |
| `tests/` | Tests `pytest` |
| `docs/` | Décisions techniques (ADR) et plans de fonctionnalités |

## Lancer le projet

```bash
pip install -r requirements.txt
make run_api      # API sur http://localhost:8000, interface sur /ui, docs sur /docs
make pytest       # tests
```

Certaines fonctions demandent un fichier `.env` (base de données, Google Cloud) qui n'est pas fourni.

## Données

- Catalogue des volcans et des éruptions : [Smithsonian Global Volcanism Program](https://volcano.si.edu/).
- Formes d'onde sismiques : centres de données FDSN publics (EarthScope, Epos-France/RESIF, GEOFON, INGV), via [ObsPy](https://docs.obspy.org/).

Les données restent la propriété de leurs fournisseurs.
