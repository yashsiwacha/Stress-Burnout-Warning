# Data Directory

This directory contains runtime data and logs for the Stress Burnout Warning System.

## Directory Structure

```
data/
├── logs/                    # Application logs
│   ├── app.log             # Main application log
│   ├── stress_data.log     # Stress monitoring data
│   └── error.log           # Error logs
│
├── user_data/              # User-specific data (privacy protected)
│   ├── stress_history.json # Historical stress data
│   ├── session_data.json   # Session information
│   └── preferences.json    # User preferences backup
│
└── cache/                  # Temporary cache files
    ├── model_cache/        # Cached model predictions
    └── ui_cache/           # UI state cache
```

## Privacy Protection

- All data in this directory is processed locally
- No personal data is transmitted to external servers
- User data is anonymized by default
- Data can be cleared through the application settings

## Data Retention

- Logs are rotated automatically to prevent excessive disk usage
- User data retention follows configured privacy settings
- Cache files are cleaned periodically

## Backup Recommendations

Consider backing up important stress history data, but ensure personal data privacy is maintained.
