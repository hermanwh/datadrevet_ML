import React, { Component } from "react";
import "./App.css";
import { Route, Link, BrowserRouter, Switch } from "react-router-dom";

import Upload from "./upload/Upload";
import CreateProject from "./createProject/CreateProject";

class App extends Component {
  render() {
    return (
      <BrowserRouter>
        <div className="Header">
          <Link to="/" style={{ textDecoration: "none", color: "white" }}>
            Datadriven machine learning
          </Link>
          <Link to="/upload" style={{ textDecoration: "none", color: "white" }}>
            New project
          </Link>
        </div>
        <Switch>
          <Route exact path="/" component={CreateProject} />
          <Route exact path="/upload" component={Upload} />
        </Switch>
      </BrowserRouter>
    );
  }
}

export default App;
