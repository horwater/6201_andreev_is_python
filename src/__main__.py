import asyncio
import sys
from src.pypline import AsyncImagePipeline

async def main(limit):
    pipeline = AsyncImagePipeline(limit)
    await pipeline.run_pipeline()

if (__name__ == '__main__'):
    if len(sys.argv) > 1:
        asyncio.run(main(int(sys.argv[1])))
    else:
        asyncio.run(main(4))