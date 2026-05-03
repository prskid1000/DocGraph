import os
import json

path = r'C:\Users\prith\.telecode\docgraph\process.py'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

old_docs_start = '        try:\n            async with session.post(f\"{base}/api/docs/add?root={slug}\",\n                                     json={\"url\": url}) as resp:\n                body = await resp.text()\n                if resp.status != 200:'
old_docs_end = '    except Exception as exc:\n        return False, f\"host route failed: {exc}\"'

d_idx_start = content.find(old_docs_start)
if d_idx_start != -1:
    d_idx_end = content.find(old_docs_end, d_idx_start) + len(old_docs_end)
    
    new_docs_logic = '''        try:
            async with session.post(f\"{base}/api/docs/add?root={slug}\",
                                     json={\"url\": url}) as resp:
                body = await resp.text()
                if resp.status != 200:
                    return False, f\"POST /api/docs/add -> HTTP {resp.status}: {body[:300]}\"
                import json
                payload = json.loads(body)
                job_id = payload.get(\"job_id\")
                if not job_id:
                    return True, payload
            
            while True:
                import asyncio
                await asyncio.sleep(2.0)
                async with session.get(f\"{base}/api/jobs/{job_id}\") as resp:
                    if resp.status != 200:
                        continue
                    job = await resp.json()
                    status = job.get(\"status\")
                    if status == \"completed\":
                        return True, job.get(\"result\") or {}
                    elif status == \"cancelled\":
                        return False, \"docs add cancelled\"
                    elif status == \"failed\":
                        return False, f\"Docs add failed: {job.get('error')}\"
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            return False, f\"host route failed: {exc}\"'''

    content = content[:d_idx_start] + new_docs_logic + content[d_idx_end:]
    print('Patched add_doc_for')

with open(path, 'w', encoding='utf-8') as f:
    f.write(content)
