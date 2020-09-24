import React from "react"
import Button from 'react-bootstrap/Button'
import axios from 'axios';

import Text from '../titles/text_display'
import "./styles/display_styles.css"

class DetDisplay extends React.Component{

  constructor(props){
    super(props)
    this.state = {
        file: this.props.file,
        det_idx: this.props.det_idx,
        pose_vector: this.props.pose_vector,
        keypoints: this.props.keypoints,
        get_retrieval_settings: this.props.get_retrieval_settings
    }
    // method used for updating the results. Comes as a prop from the Root App component
    this.update_results = this.props.update_results.bind(this)

    this.selectInstance = this.selectInstance.bind(this)
  }

  async selectInstance(){
    // creating an object to send to API via pose
    const formData = new FormData()
    const skip = ["file"]
    formData.append("timestamp", new Date().toLocaleString())
    for (var name in this.props) {
      if(skip.includes(name)){
        continue
      }
      formData.append(name, this.props[name]);
    }
    // adding the retrueval settings read from the input are
    var retrieval_settings = this.props.get_retrieval_settings()
    for (var setting in retrieval_settings) {
      formData.append(setting, retrieval_settings[setting]);
    }

    var results = undefined;
    // establishing connection, sending and awaiting response
    axios({
      method: 'post',
      url: 'http://localhost:5000/api/retrieve/',
      data: formData,
      headers: {'content-type': 'multipart/form-data',
                "Accept": "application/json"}
    })
    .then(function (response) {
        //handle success
        console.log("Success")
        results = response.data
    })
    .catch(function (response) {
        //handle error
        console.log("Error!")
        results = 0
    })
    .finally(() => {
      // logic executed after having received the response
      console.log("Finally")
      if(results !== 0){
        this.update_results(results)
      }
    });

  }


  render(){
    var title = "Detection " + this.state.det_idx
    var img = URL.createObjectURL(this.props.file)
    return(
      <div className="img_display_area">
        <div>
          <Text text_display={title} font_size="1.4em" font_style="normal"/>
        </div>
        <div className="det_display" style={{backgroundImage: "url("+img+")"}}></div>
        <Button className="myButton" variant="primary" onClick={this.selectInstance}>Select</Button>
      </div>
    )
  }

}

export default DetDisplay

//
