# DAT640 MusicCRS — Part 4 Proposal
## Adaptive Music Recommendations Using Relevance Feedback

### 1. Objective and Motivation

The objective of this feature is to improve the MusicCRS recommendation system by allowing it to adapt to user preferences through conversational feedback.

The existing system retrieves and ranks tracks based on a user's initial request or playlist. However, user preferences may change or become more specific throughout a conversation. A user might enjoy some recommended tracks while rejecting others, or request adjustments such as "more energetic songs" or "less heavy metal."

We propose implementing an adaptive recommendation mechanism that incorporates positive and negative feedback into subsequent retrieval and ranking. This would make the system more personalized, interactive, and responsive to users' preferences.

### 2. Proposed Functionality

The system will support the following advanced features:

1. **Natural-language feedback:** Users can express positive or negative feedback about recommended tracks, artists, or musical characteristics.
2. **Preference tracking:** The system maintains a session-level preference profile based on the user's feedback.
3. **Feedback-aware ranking:** Subsequent recommendations are re-ranked according to the updated preferences, rewarding preferred characteristics and penalizing rejected ones.
4. **Iterative refinement:** Users can provide feedback over multiple conversation turns, allowing recommendations to improve progressively.
5. **Transparent adaptation:** The system can briefly explain how feedback affected subsequent recommendations.

Example interaction:

- **User:** "Recommend some energetic rock songs for working out."
- **System:** Recommends five relevant tracks.
- **User:** "I like the first two, but the others are too heavy. I'd prefer more alternative rock."
- **System:** Updates the preference profile and produces a new set of recommendations with less emphasis on heavy metal and greater emphasis on alternative rock.

### 3. Technical Approach

The implementation will build on the existing MusicCRS retrieval and ranking component developed in Part 1.

**Feedback interpretation:** Natural-language feedback will be interpreted into structured positive and negative preferences, using the existing conversational framework and, where necessary, an LLM.

**Preference representation:** Preferences will be maintained in a session-level profile containing positively and negatively weighted attributes, such as artists, genres, and other available track metadata.

**Relevance feedback and query expansion:** Positive feedback may be used to expand the original query with preferred characteristics, while negative feedback may reduce the importance of unwanted characteristics.

**Feedback-aware re-ranking:** Candidate tracks will be retrieved using the existing retrieval component and re-ranked using a combined scoring function:

\[
S(d,q,u)=\alpha S_{\text{retrieval}}(d,q)+\beta S_{\text{preference}}(d,u)
\]

where \(S_{\text{retrieval}}\) represents the original relevance score and \(S_{\text{preference}}\) measures agreement with the user's feedback profile. Scores will be normalized as necessary.

The implementation will use track attributes available in the TalkPlay dataset. If some musical characteristics are unavailable, preference extraction and ranking will be limited to supported metadata or existing representations.

### 4. Evaluation Methodology

The feature will be evaluated using predefined conversational test scenarios and comparisons against the existing recommendation system.

Evaluation will include:

- Correct interpretation of positive and negative feedback.
- Consistent preservation of preferences across multiple conversation turns.
- Observable changes in recommendation rankings after feedback.
- Reduced occurrence of explicitly rejected attributes in subsequent recommendations.
- Preservation of relevance to the original user request.
- Response time within the project's 3–5-second requirement.

The evaluation will compare recommendations before and after feedback using controlled examples and available metadata.

### 5. Proposed Grading Rubric

| Requirement | Points |
|---|---:|
| Correctly interpret positive and negative feedback expressed in natural language | 3 |
| Maintain and update user preferences across multiple conversation turns | 3 |
| Implement feedback-aware retrieval or re-ranking that changes subsequent recommendations | 4 |
| Demonstrate and evaluate improvements using predefined test scenarios | 3 |
| Maintain response times within the required limits and respect the LLM call budget | 2 |
| **Total** | **15** |

### 6. Expected Outcome

The proposed feature will extend MusicCRS from a primarily request-based recommendation system into an adaptive conversational recommender.

By applying relevance feedback, query expansion, and re-ranking techniques from DAT640, the system will be able to respond more effectively to evolving user preferences while remaining compatible with the existing application architecture.


# DAT640 MusicCRS — Part 4 Proposal
## Diversity-Aware Playlist Generation and Re-Ranking

### 1. Objective and Motivation

The objective of this feature is to improve playlist quality by balancing recommendation relevance with musical diversity.

Traditional retrieval and ranking approaches prioritize tracks that best match a user's query. Although this can produce highly relevant results, the resulting playlist may contain excessive repetition, such as multiple tracks from the same artist or closely related musical categories.

We propose implementing a diversity-aware re-ranking mechanism that produces more varied playlists while maintaining relevance to the user's request.

This would improve the user experience by introducing variety, reducing repetitive recommendations, and allowing users to influence the balance between familiarity and exploration.

### 2. Proposed Functionality

The system will provide the following advanced features:

1. **Diversity-aware re-ranking:** Candidate tracks will be re-ranked to balance relevance and similarity to already selected tracks.
2. **Artist and musical diversity:** The system will reduce excessive repetition of artists and, where supported by available metadata, genres or other musical characteristics.
3. **User-controlled diversity:** Users can request more familiar or more varied recommendations using natural language.
4. **Diversity-aware playlist generation:** The system can generate playlists that maintain a coherent theme while providing a broader selection of tracks.
5. **Relevance preservation:** Diversity improvements should not substantially reduce the relevance of recommendations to the original request.

Example interaction:

- **User:** "Create an energetic workout playlist."
- **System:** Generates a playlist of relevant tracks with diversity-aware re-ranking.
- **User:** "Make it more diverse. I want fewer songs from the same artists."
- **System:** Re-ranks or regenerates the playlist to increase artist variety while maintaining the energetic workout theme.

### 3. Technical Approach

The implementation will extend the existing retrieval component developed in Part 1 by introducing a diversity-aware re-ranking stage.

**Candidate retrieval:** The existing retrieval model will generate a ranked list of candidate tracks relevant to the user's request.

**Similarity measurement:** Track similarity will be estimated using available metadata, such as artist and genre, or existing track representations.

**Maximal Marginal Relevance (MMR):** The system will use an MMR-inspired approach to iteratively select tracks that balance query relevance and novelty relative to tracks already selected:

\[
MMR(d)=\lambda Rel(d,q)-(1-\lambda)\max_{s\in S}Sim(d,s)
\]

where:

- \(Rel(d,q)\) measures relevance to the user request.
- \(Sim(d,s)\) measures similarity between a candidate and an already selected track.
- \(S\) is the set of previously selected tracks.
- \(\lambda\) controls the relevance-diversity trade-off.

The system will prioritize relevance when the user requests a focused playlist and increase the diversity weight when the user requests more variety.

The similarity function and supported diversity dimensions will depend on the metadata available in the TalkPlay dataset.

### 4. Evaluation Methodology

The feature will be evaluated by comparing standard relevance-based ranking against diversity-aware re-ranking.

Evaluation will include:

- **Artist diversity:** Number of distinct artists relative to playlist length.
- **Intra-list diversity:** Average dissimilarity between recommended tracks using the selected similarity representation.
- **Relevance preservation:** Comparison of retrieval relevance before and after re-ranking.
- **User control:** Verification that natural-language diversity requests modify the resulting recommendations.
- **Response time:** Confirmation that re-ranking satisfies the project's latency requirement.

Multiple predefined queries and playlist scenarios will be used to assess whether the proposed approach increases variety without excessively reducing relevance.

### 5. Proposed Grading Rubric

| Requirement | Points |
|---|---:|
| Implement diversity-aware re-ranking using MMR or an equivalent algorithm | 4 |
| Support artist diversity and at least one additional diversity dimension where metadata permits | 3 |
| Allow users to adjust diversity through natural-language interaction | 2 |
| Evaluate diversity and relevance against the existing retrieval baseline | 4 |
| Maintain response times within the required limits and respect the LLM call budget | 2 |
| **Total** | **15** |

### 6. Expected Outcome

The proposed feature will extend MusicCRS with a ranking strategy that considers both relevance and diversity.

By applying similarity measurement, re-ranking, and retrieval evaluation techniques from DAT640, the system will generate playlists that are relevant, less repetitive, and better suited to users seeking musical variety.

The approach will reuse the existing retrieval component and introduce a separate re-ranking stage, making it compatible with the current MusicCRS architecture.