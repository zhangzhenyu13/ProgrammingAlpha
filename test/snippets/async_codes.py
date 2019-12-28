import random
import asyncio


potatos=[]

class Potato:
    @classmethod
    def make(cls, num):
        potatos=[]
        for _ in range(num):
            potatos.append(cls())
        
        return potatos

async def ask_for_ptotatos():
    num=random.randint(2,10)
    await asyncio.sleep(random.random())
    potatos.extend(Potato.make(num))

async def take_potatos(num):
    count=0
    while count<num:
        if len(potatos)==0:
            await ask_for_ptotatos()
        potato=potatos.pop()
        yield potato
        count+=1

async def buy_potatos():
    bucket=[]
    async for potato in take_potatos(50):
        bucket.append(potato)
        print("potato:",id(potato), "bought" )

async def buy_tomatos():
    bucket=[]
    async for potato in take_potatos(50):
        bucket.append(potato)
        print("tomato:",id(potato), "bought" )

if __name__ == "__main__":
    loop=asyncio.get_event_loop()
    loop.run_until_complete(asyncio.wait([buy_potatos(), buy_tomatos()]))
    loop.close()