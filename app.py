from src.Model.models import train
from src.HtmlLayout.index import app




if __name__ == '__main__':
    # Modelle
    train()
    
    app.run()