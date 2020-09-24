import React from "react"

import Text from '../../../components/text_display/text_display.js'
import "./styles/display_styles.css"

class ResultDisplay extends React.Component{

  render(){
    var title = "Retrieval #" + this.props.det_idx + "%aux%" +
                "Metric: " + Math.round(this.props.metric * 100) / 100
    var img = URL.createObjectURL(this.props.file)
    return(
      <div className="img_display_area result">
        <div>
          <Text text_display={title} font_size="1.0em" font_style="normal"/>
        </div>
        <div className="result_display" style={{backgroundImage: "url("+img+")"}}></div>
      </div>
    )
  }

}

export default ResultDisplay

//
