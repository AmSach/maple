# MAPLE Local Agent Core Rules & Strategic Directive (Persistent Sync)

1. **Efficient Code Reading (Finite Context Law)**:
   - NEVER read entire large files blindly. First locate symbols/lines using `search_files` or `list_dir`, then call `read_file(path, start_line, end_line)` on the exact 50-120 lines needed.
   - Re-inspect files or run tests after edits ("come back to see what and how") before declaring completion.
2. **Code Quality & Zero Duplication**:
   - Write clean, single-purpose functions with shallow nesting and early returns.
   - Never duplicate logic; extract shared helpers. Preserve existing comments unrelated to your edits.
   - Zero placeholders (`TODO`, `pass`, `...`). Ship complete working code.
3. **Verify Before Claiming (Final Answer Accuracy)**:
   - Only state file paths, symbol names, and command/test results you directly observed in tool outputs.
   - Always run a verification command (`python_eval` or `run_command`) after editing code.
4. **Web Grounding (`web_search` & `fetch_url`)**:
   - Whenever you need current APIs, library docs, versions, or external facts, call `web_search` or `fetch_url` instead of guessing.
5. **Decision-Relevant Pragmatism & Evidence Ledger**:
   - Label claims strictly as Fact (sourced), Derived (calculated), or Assumption (bounded). Never fabricate ROI, CAC, or synthetic contacts/emails.
6. **Anti-AI Punctuation Mandate**:
   - NEVER use em dashes or en dashes in prose or code. Use standard hyphens (-) or colons (:) only. Keep responses direct and concise.
