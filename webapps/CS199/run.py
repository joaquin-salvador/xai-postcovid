# webapps/CS199/run.py
from shapash_dashboard.app import create_app

if __name__ == '__main__':
    app = create_app()
    app.run_server(debug=True, port=8050)