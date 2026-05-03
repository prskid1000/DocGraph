import os
import re

path = r'C:\Users\prith\.telecode\docgraph\process.py'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

old_docs = r'''            async with session.post\(f\"\{base\}/api/docs/add\?root=\{slug\}\",\s+json={\"url\": url}\) as resp:\s+body = await resp.text\(\)\s+if resp.status != 200:\s+return False, f\"POST /api/docs/add [^\"]+: \{body\[:300\]\}\"\s+try:\s+return True, json.loads\(body\)\s+except Exception:\s+return True, \{\"raw\": body\[:1000\]\}\s+except Exception as exc:\s+return False, f\"host route failed: \{exc\}\"'''

new_docs = '''            async with session.post(f\"{base}/api/docs/add?root={slug}\",
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

content = re.sub(old_docs, new_docs, content)

with open(path, 'w', encoding='utf-8') as f:
    f.write(content)
print('Patched add_doc_for with regex')
