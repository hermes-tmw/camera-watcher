# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Architecture Overview

This is a multi-service camera surveillance system that processes motion detection video clips using machine learning. The system is built around a Flask API with Redis queues for distributed processing.

### Core Components

- **Motion Detection**: Uses Motion Project binary to capture video clips when motion is detected
- **API Server**: Flask application (`api.py`) providing REST endpoints for event management and labeling
- **Worker Services**: Distributed task processing via Redis/RQ queues
  - `ioworker`: Handles file I/O operations and basic event recording
  - `videoworker`: Processes video files to extract significant frames
  - `predictionworker`: Runs ML predictions on extracted frames using FastAI models
- **Database**: SQLAlchemy models with support for event observations, classifications, and labeling

### Key Data Flow

1. Motion detection creates video files and triggers events
2. Events are queued for processing through Redis
3. Video worker extracts significant frames from clips
4. Prediction worker runs ML classification on frames
5. Results are stored and made available through API for manual labeling

### Database Schema

Core models in `watcher/model.py`:
- `EventObservation`: Main event record with video file, timing, and metadata
- `Labeling`: Human classifications with support for multiple labels
- `IntermediateResult`: ML prediction results and extracted frames
- `Computation`: Legacy prediction tracking

## Development Commands

### Docker Operations
```bash
# Start all services
docker compose up

# Start with specific profiles
docker compose --profile web up     # API + nginx only
docker compose --profile all up     # All services including predictor

# Build services
docker compose build
```

### Testing
```bash
# Run test suite (uses unittest framework)
python -m unittest discover tests/

# Run specific test file
python -m unittest tests.test_api

# Run API in CLI mode for testing
python api.py dbtest
python api.py get_uncategorized
```

### Utility Commands via watchutil.py
```bash
# Worker processes (run via Docker in production)
./watchutil.py ioworker          # Process event recording queue
./watchutil.py videoworker       # Process video frame extraction
./watchutil.py predictionworker  # Process ML predictions

# Data management
./watchutil.py update_lighting   # Update lighting metadata for events
./watchutil.py syncup           # Sync data to remote server
./watchutil.py uncategorized    # Show unlabeled events
./watchutil.py set_user <username>  # Generate API key for user

# Queue management
./watchutil.py failed           # Show failed queue jobs
./watchutil.py failed purge     # Clear failed jobs

# Migration utilities
./watchutil.py migrate-labels   # Migrate old classification format
./watchutil.py migrate-stills   # Migrate computation results
```

### Configuration

- Environment variables: `WATCHER_CONFIG`, `WATCHER_LOG_FILE`, `REDIS_URL`
- Config location: `/usr/local/etc/watcher.cfg` (in containers)
- Database URL configured via `watcher.connection.get_db_url()`

### Key Directories

- `data/video/`: Video file storage organized by camera/date
- `watcher/`: Core Python package with models and processing logic
- `etc/`: Configuration files for services (nginx, motion, redis)
- `log/`: Application and service logs
- `tests/`: Unit tests using unittest framework

## Testing Notes

- Tests use in-memory SQLite for isolation
- Test utilities in `watcher/tests/utils.py` for database setup
- API tests include authentication and endpoint validation
- Uses unittest framework, not pytest