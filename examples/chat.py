import typer
from amp import Amp
from rich.console import Console
from rich.prompt import Prompt

console = Console()
app = typer.Typer()

@app.command()
def chat():
    """
    Interactive Chat with AMP Memory.
    Everything you say is remembered.
    """
    brain = Amp() # Default to ~/.amp/brain.db
    console.print("[bold green]Welcome to AMP Chat![/bold green]")
    console.print("Type 'exit' to quit. Type 'consolidate' to force move STM -> LTM.\n")

    # 1. Fetch recent context
    console.print("[dim]Fetching recent memories...[/dim]")
    # We can use recall with a generic query or just peek at recent
    recents = brain.recall("current context", limit=3)
    if recents:
        console.print("[bold blue]Context Loaded:[/bold blue]")
        for r in recents:
            console.print(f" - {r['content']}")
    else:
        console.print("[dim]No relevant context found.[/dim]")

    while True:
        user_input = Prompt.ask("\n[bold yellow]You[/bold yellow]")
        
        if user_input.lower() in ["exit", "quit"]:
            break
            
        if user_input.lower() == "consolidate":
            count = brain.consolidate()
            console.print(f"[bold green]Consolidated {count} memories to LTM.[/bold green]")
            continue

        # 2. Search for relevant memories *before* answering (RAG)
        hits = brain.recall(user_input, limit=2)
        context_str = ""
        if hits:
            console.print("[dim]Recall triggered:[/dim]")
            for h in hits:
                if h['score'] > 0.6: # Only show high relevance
                    console.print(f" [dim]Remembered: {h['content']} ({h['score']:.2f})[/dim]")
                    context_str += f"- {h['content']}\n"
        
        # 3. Simulate an "Agent" response (using the context)
        # In a real app, this would go to an LLM.
        response = f"I heard you say: '{user_input}'."
        if context_str:
            response += f"\nI also remember:\n{context_str}"
            
        console.print(f"[bold cyan]AMP[/bold cyan]: {response}")

        # 4. Store the interaction
        brain.remember(f"User said: {user_input}")

    console.print("Goodbye!")

if __name__ == "__main__":
    app()
