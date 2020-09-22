import React from 'react';
import { makeStyles } from '@material-ui/core/styles';
import Card from '@material-ui/core/Card';
import CardActionArea from '@material-ui/core/CardActionArea';
import CardActions from '@material-ui/core/CardActions';
import CardContent from '@material-ui/core/CardContent';
import CardMedia from '@material-ui/core/CardMedia';
import Button from '@material-ui/core/Button';
import Typography from '@material-ui/core/Typography';
import { Link } from "react-router-dom";


class CustomCard extends React.Component{

  render(){

    return (
      <Link to={this.props.route}>
        <Card style={{maxWidth: 345}}>
          <CardActionArea>
            <CardMedia style={{height: 200, objectFit: "cover"}}>
              <img src={this.props.image} style={{objectFit: "cover", height: "100%",
                  width: "100%", objectPosition: "0% 40%"}}/>
            </CardMedia>
            <CardContent>
              <Typography gutterBottom variant="h5" component="h2">
                {this.props.title}
              </Typography>
              <Typography variant="body2" color="textSecondary" component="p">
                {this.props.description}
              </Typography>
            </CardContent>
          </CardActionArea>

          {/* Uncomment to add buttons at the bottom of the card
            <CardActions>
             <Button size="large" color="primary">
               Try it!
             </Button>
          </CardActions>*/}
        </Card>
      </Link>
    );
  }
}

export default CustomCard;


//
