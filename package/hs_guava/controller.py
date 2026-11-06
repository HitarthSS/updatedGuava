"""Authors' controller registration boundary. Each execution is one tool decision."""


class AuthorController:
    async def run(self, ctx):
        while not ctx.ended:
            action = await ctx.policy.act()
            await ctx.execute(action)
