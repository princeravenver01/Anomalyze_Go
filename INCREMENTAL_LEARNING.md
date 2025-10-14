# Incremental Learning Feature 🤖

## Overview

Anomalyze now includes **automatic incremental learning** - the system saves every uploaded network log file and automatically retrains the model when enough new data is collected.

## How It Works

### 1. **Upload & Save** 📤

- When a user uploads network logs via the web interface
- The file is immediately analyzed for anomalies (as before)
- **NEW**: System checks if file is a duplicate using SHA-256 hash
- **If unique**: File is saved to `data/uploaded_logs/` with timestamp
- **If duplicate**: File is skipped, no save occurs
- Files are timestamped: `20251014_103045_upload.txt`

### 2. **Upload Tracking** 📊

- System tracks number of **unique** uploads in `models/upload_counter.txt`
- Counter increments only for new, unique files
- Duplicate uploads don't increment the counter
- Default threshold: **10 unique uploads** triggers retraining

### 3. **Automatic Retraining** 🔄

- When threshold is reached, `retrain_model.py` runs automatically
- Process runs in background (non-blocking)
- Combines original training data + all uploaded logs
- Retrains all 5 ensemble K-means models
- Optimizes detection threshold
- Saves new models, replacing old ones

### 4. **Archiving** 📦

- After successful retraining, uploaded logs are archived
- Moved to `data/uploaded_logs/archived/`
- Files are timestamped for tracking
- Counter resets to 0

## File Structure

```
data/
├── uploaded_logs/
│   ├── .gitkeep
│   ├── 20251014_103045_upload1.txt  (pending)
│   ├── 20251014_104522_upload2.txt  (pending)
│   └── archived/
│       └── 20251014_110000_20251014_103045_upload1.txt
models/
└── upload_counter.txt  (tracks upload count)
```

## Duplicate Detection 🔍

The system automatically detects and skips duplicate uploads:

### How It Works

- **SHA-256 Hash**: Each file is hashed upon upload
- **Comparison**: Hash is compared against all existing files in `uploaded_logs/`
- **Match Found**: File is marked as duplicate, not saved again
- **No Match**: File is saved as new, counter increments

### Benefits

✅ **Prevents redundant storage** - Same file isn't saved multiple times  
✅ **Accurate counting** - Only unique files count toward retraining threshold  
✅ **User feedback** - API response indicates if file was duplicate  
✅ **Fast comparison** - Hash-based comparison is efficient

### API Response

```json
{
  "success": true,
  "duplicate_file": false,
  "upload_counter": 5,
  "anomalies": [...],
  "metrics": {...}
}
```

If duplicate detected:

```json
{
  "success": true,
  "duplicate_file": true,
  "upload_counter": null,
  "anomalies": [...],
  "metrics": {...}
}
```

## Configuration

You can adjust the retraining threshold in `api_server.py`:

```python
RETRAIN_THRESHOLD = 10  # Retrain after N unique uploads
```

**Recommendations:**

- **Low traffic**: 5-10 uploads
- **Medium traffic**: 20-50 uploads
- **High traffic**: 100+ uploads

## Manual Retraining

You can also manually trigger retraining at any time:

```bash
python retrain_model.py
```

This will:

1. Combine all training data
2. Retrain ensemble models
3. Optimize threshold
4. Save new models
5. Archive uploaded logs

## Monitoring

Check the API logs to see:

- Upload counter: `Step 1.6: Upload count: 3/10`
- Retraining trigger: `Step 1.7: Retraining threshold reached`
- Retraining status: `✓ Retraining initiated (PID: 12345)`

## Benefits

✅ **Continuous Improvement**: Model adapts to new traffic patterns  
✅ **Automatic**: No manual intervention required  
✅ **Non-Blocking**: Retraining happens in background  
✅ **Trackable**: All data is timestamped and archived  
✅ **Configurable**: Adjust thresholds to your needs

## API Behavior

### During Retraining

- ✅ API continues serving predictions with old model
- ✅ New uploads continue to be saved
- ✅ Once complete, API automatically loads new model on next prediction

### Upload Counter API

```bash
# Check current upload count
curl https://your-api-url.com/api/model-info
```

Returns:

```json
{
  "num_models": 5,
  "num_features": 122,
  "optimal_threshold": 0.123456,
  "model_type": "K-Means Ensemble"
}
```

## Deployment Notes

### For Render (Backend API)

- Retraining works automatically
- Ensure sufficient disk space for uploaded logs
- Consider memory limits for large retraining jobs

### For Vercel (Frontend)

- No changes needed
- Frontend forwards uploads to Render API
- Incremental learning happens on backend

## Troubleshooting

**Issue**: Retraining doesn't trigger  
**Solution**: Check `models/upload_counter.txt` - verify it's incrementing

**Issue**: Out of disk space  
**Solution**: Lower `RETRAIN_THRESHOLD` or manually delete archived files

**Issue**: Retraining takes too long  
**Solution**: Increase `RETRAIN_THRESHOLD` or reduce training data size

## Future Enhancements

Potential improvements:

- [ ] Add retraining status endpoint
- [ ] Email notifications on retraining completion
- [ ] Schedule-based retraining (daily/weekly)
- [ ] Web dashboard for upload/retrain monitoring
- [ ] Model versioning and rollback capability

---

**Last Updated**: October 14, 2025  
**Status**: ✅ Active and Deployed
