import React from "react"
import Button from 'react-bootstrap/Button'

import Text from '../titles/text_display'
import "./styles/display_styles.css"

class DetDisplay extends React.Component{

  constructor(props){
    super(props)
    this.state = {
        file: this.props.file,
        det_idx: this.props.det_idx,
        pose_vector: this.props.pose_vector,
        keypoint_vector: this.props.keypoint_vector
    }
    this.selectThisDet = this.selectThisDet.bind(this)
  }

  selectThisDet(){
    var a = 0
  }


  render(){
    var title = "Detection " + this.state.det_idx
    var img = URL.createObjectURL(this.props.file)
    return(
      <div className="img_display_area">
        <div>
          <Text text_display={title} font_size="1.6em" font_style="normal"/>
        </div>
        <div className="det_display" style={{backgroundImage: "url("+img+")"}}></div>
        <Button className="myButton" variant="primary" onClick="selectThisDet">Select</Button>
      </div>
    )
  }

}

export default DetDisplay

//
