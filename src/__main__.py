import asyncio
import sys
from src.pypline import AsyncImagePipeline

def main():
    if len(sys.argv) > 1:
        limit = int(sys.argv[1])
    else:
        limit = 2
    asyncio.run(run_main(limit))

async def run_main(limit):
    """Асинхронная часть пайплайна"""
    pipeline = AsyncImagePipeline(limit)
    await pipeline.run_pipeline()

if __name__ == '__main__':
    main()