import asyncio
from vare.demo import run_demo


async def main():
    results = await run_demo(rounds=8, rollouts=512, seed=7)
    for r in results:
        print(r)


if __name__ == "__main__":
    asyncio.run(main())
