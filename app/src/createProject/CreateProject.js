import React, { Component } from "react";
import "./CreateProject.css";
import Upload from "../upload/Upload";
import DownloadFile from "../download/Download";

class CreateProject extends Component {
  render() {
    return (
      <React.Fragment>
        <div className="Subheader">
          Upload a tensorflow model (with weights) or use predefined model
        </div>
        <div className="Container">
          <div className="Card">
            <DownloadFile />
          </div>
          <div className="Card">
            <Upload />
          </div>
        </div>
      </React.Fragment>
    );
  }
}

export default CreateProject;
