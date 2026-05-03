import os
import json

path = r'C:\Users\prith\.telecode\docgraph\process.py'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

old_index_start = '        try:\n            try:\n                async with session.post(url, json={\"full\": full}) as resp:\n                    body = await resp.text()\n                    if resp.status == 499:'
old_index_end = '            except Exception as exc:\n                return False, f\"POST /api/admin/index failed: {exc}\"'

idx_start = content.find(old_index_start)
if idx_start != -1:
    idx_end = content.find(old_index_end, idx_start) + len(old_index_end)
    
    new_index_logic = '''        try:
            try:
                async with session.post(url, json={\"full\": full}) as resp:
                    body = await resp.text()
                    if resp.status != 200:
                        return False, f\"POST /api/admin/index -> HTTP {resp.status}: {body[:500]}\"
                    import json
                    payload = json.loads(body)
                    job_id = payload.get(\"job_id\")
                    if not job_id:
                        return False, \"No job_id returned\"
                
                while True:
                    await asyncio.sleep(2.0)
                    async with session.get(f\"{base}/api/jobs/{job_id}\") as resp:
                        if resp.status != 200:
                            continue
                        job = await resp.json()
                        status = job.get(\"status\")
                        if status == \"completed\":
                            stats = job.get(\"result\") or {}
                            cap = job.get(\"log\") or \"\"
                            lines: list[str] = []
                            if cap:
                                lines.append(cap.rstrip())
                            summary = (
                                f\"\\n--- done: {stats.get('files', '?')} files, \"
                                f\"{stats.get('changed', '?')} changed, \"
                                f\"{stats.get('deleted', '?')} deleted, \"
                                f\"{stats.get('entities', '?')} entities, \"
                                f\"{stats.get('errors', 0)} errors, \"
                                f\"{stats.get('elapsed', 0):.2f}s ---\"
                            )
                            lines.append(summary)
                            return True, \"\\n\".join(lines)
                        elif status == \"cancelled\":
                            raise asyncio.CancelledError()
                        elif status == \"failed\":
                            return False, f\"Index job failed: {job.get('error')}\"
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                return False, f\"POST /api/admin/index failed: {exc}\"'''
    
    content = content[:idx_start] + new_index_logic + content[idx_end:]
    print('Patched _index_via_host')

old_wiki_start = '        try:\n            try:\n                async with session.post(url, json={\"force\": force}) as resp:\n                    body = await resp.text()\n                    if resp.status == 499:'
old_wiki_end = '            except Exception as exc:\n                return False, f\"POST /api/wiki/build failed: {exc}\"'

w_idx_start = content.find(old_wiki_start)
if w_idx_start != -1:
    w_idx_end = content.find(old_wiki_end, w_idx_start) + len(old_wiki_end)
    
    new_wiki_logic = '''        try:
            try:
                async with session.post(url, json={\"force\": force}) as resp:
                    body = await resp.text()
                    if resp.status != 200:
                        return False, f\"POST /api/wiki/build -> HTTP {resp.status}: {body[:500]}\"
                    import json
                    payload = json.loads(body)
                    job_id = payload.get(\"job_id\")
                    if not job_id:
                        return False, \"No job_id returned\"
                
                while True:
                    await asyncio.sleep(2.0)
                    async with session.get(f\"{base}/api/jobs/{job_id}\") as resp:
                        if resp.status != 200:
                            continue
                        job = await resp.json()
                        status = job.get(\"status\")
                        if status == \"completed\":
                            res = job.get(\"result\") or {}
                            built = res.get(\"built\", \"?\")
                            modules = res.get(\"modules\") or []
                            status_str = 'rebuilt' if force else 'resumable'
                            summary = (
                                f\"\\n--- wiki: {built} module page(s) \"
                                f\"({status_str}) ---\\n\"
                                + \"\\n\".join(f\"  * {m}\" for m in modules[:50])
                                + (\"\\n  * ...\" if len(modules) > 50 else \"\")
                            )
                            return True, summary
                        elif status == \"cancelled\":
                            raise asyncio.CancelledError()
                        elif status == \"failed\":
                            return False, f\"Wiki job failed: {job.get('error')}\"
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                return False, f\"POST /api/wiki/build failed: {exc}\"'''
    
    content = content[:w_idx_start] + new_wiki_logic + content[w_idx_end:]
    print('Patched _wiki_via_host')

with open(path, 'w', encoding='utf-8') as f:
    f.write(content)
