# TODO - RumorDetection Improvements

## Task: Fix Indian News Classification + Add Real-Time Fact-Checking

### Phase 1: Add Real-Time Google News Fact-Check Feature
- [ ] 1.1 Create a new API endpoint `/api/factcheck` that:
  - Takes a news headline/text as input
  - Searches Google News for the headline
  - Analyzes the sources from search results
  - Returns credibility assessment based on known reliable sources
- [ ] 1.2 Add source credibility database (reliable news sources)
- [ ] 1.3 Integrate fact-check sites (Snopes, FactCheck.org, PIB India, Boom)
- [ ] 1.4 Update UI to show fact-check results

### Phase 2: Improve India-Specific Model
- [ ] 2.1 Analyze current India dataset limitations
- [ ] 2.2 Create enhanced India dataset with more samples
- [ ] 2.3 Retrain model with balanced India data
- [ ] 2.4 Update train_india_model.py with better parameters

### Phase 3: Testing & Integration
- [ ] 3.1 Test real-time fact-check feature
- [ ] 3.2 Test India model accuracy
- [ ] 3.3 Integrate both features in the main app
