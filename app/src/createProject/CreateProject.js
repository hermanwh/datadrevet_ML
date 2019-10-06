import React, { Component } from "react";
import "./CreateProject.css";
import DownloadFile from "../download/Download";

class CreateProject extends Component {
  render() {
    return (
      <React.Fragment>
        <div className="Subheader">
          Use predefined model or upload your own model at "new project"
        </div>
        <DownloadFile />
      </React.Fragment>
    );
  }
}

export default CreateProject;
